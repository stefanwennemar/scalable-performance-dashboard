"""Portfolio value time series and performance/risk metrics.

The value series is built by replaying every security transaction day-by-day
against a daily price panel (one column per ISIN). For every day we know:
    holdings_t = matrix of shares per ISIN at end of day t
    cash_t = cash balance at end of day t
    value_t = holdings_t · prices_t + cash_t

External cash flows (deposits, withdrawals, cash transfers) are tracked
separately so we can compute true Time-Weighted Returns (TWR), a Modified
Dietz approximation, simple money-on-money returns, and XIRR.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq


BERLIN_TZ = ZoneInfo("Europe/Berlin")


def effective_today() -> pd.Timestamp:
    """Today's calendar date in Europe/Berlin.

    All displayed dates follow Berlin's calendar — so the moment Berlin
    rolls past midnight, the dashboard considers the new day as "today",
    no matter what time zone the dashboard is being run from. This is
    important e.g. for a user in PDT checking at 23:30 local (= 08:30
    Berlin next day): the dashboard will already show the new Berlin date
    even though Xetra hasn't opened yet. Numbers will then update as
    gettex starts publishing the new day's quotes.
    """
    return pd.Timestamp(datetime.now(BERLIN_TZ).date())

EXTERNAL_CASH_TYPES = {
    "Deposit", "Withdrawal", "Cash Transfer In", "Cash Transfer Out",
}


@dataclass
class ValuePanel:
    dates: pd.DatetimeIndex      # daily
    holdings: pd.DataFrame       # rows=dates, cols=ISIN  (shares, end-of-day)
    cash: pd.Series              # rows=dates (EUR end-of-day)
    prices: pd.DataFrame         # rows=dates, cols=ISIN  (EUR)
    value: pd.Series             # total portfolio value, EUR end-of-day
    external_flow: pd.Series     # net external cash flow on that day (+ = inflow,
                                 #   includes capital-gains tax as a negative flow)
    tax_paid: pd.Series          # capital-gains tax paid that day (positive number)


def build_value_panel(transactions: pd.DataFrame, prices: pd.DataFrame,
                      live_prices: dict[str, float] | None = None,
                      ) -> ValuePanel:
    """Replay transactions chronologically to produce a daily value panel."""
    tx = transactions.sort_values("datetime").reset_index(drop=True)
    start = pd.Timestamp(tx["datetime"].min().date())
    end = effective_today()
    dates = pd.date_range(start, end, freq="D")

    isins = sorted({i for i in tx["isin"].dropna().unique()})

    # End-of-day holdings: cumulative net shares per ISIN.
    holdings = pd.DataFrame(0.0, index=dates, columns=isins)
    cash = pd.Series(0.0, index=dates)
    external = pd.Series(0.0, index=dates)

    # Build daily deltas first, then cumsum.
    shares_delta = pd.DataFrame(0.0, index=dates, columns=isins)
    cash_delta = pd.Series(0.0, index=dates)
    tax_paid = pd.Series(0.0, index=dates)

    for row in tx.itertuples(index=False):
        d = pd.Timestamp(row.datetime.date())
        if d not in dates:
            continue
        ttype = row.type
        if ttype == "Taxes":
            continue
        amount = row.amount if pd.notna(row.amount) else 0.0
        shares = row.shares if pd.notna(row.shares) else 0.0

        if ttype == "Taxes":
            # Tax refund/charge: real cash movement, but excluded from
            # performance — model as an external flow.
            if row.assetType == "Cash" and amount != 0:
                cash_delta.loc[d] += amount
                external.loc[d] += amount
            continue

        if row.assetType == "Cash":
            fee = row.fee if pd.notna(row.fee) else 0.0
            if ttype == "Distribution":
                # Gross of withholding; cash gets the net. Tax outflow is
                # treated as external so it doesn't depress performance.
                tax = row.tax if pd.notna(row.tax) else 0.0
                cash_delta.loc[d] += (amount - tax)
                if tax != 0:
                    external.loc[d] -= tax
                    tax_paid.loc[d] += tax
            else:
                cash_delta.loc[d] += (amount - fee)
                if ttype in EXTERNAL_CASH_TYPES:
                    external.loc[d] += amount  # the fee, if any, is a real cost
            continue

        isin = row.isin if pd.notna(row.isin) else None
        if not isin:
            continue

        if ttype in ("Buy", "Savings plan", "Reinvestment_Distribution"):
            shares_delta.loc[d, isin] += shares
            fee = row.fee if pd.notna(row.fee) else 0.0
            cash_delta.loc[d] += (amount - fee)
        elif ttype == "Sell":
            shares_delta.loc[d, isin] -= shares
            # Sell proceeds: net of capital-gains tax paid at source.
            # The tax outflow is modelled as an external flow (so it doesn't
            # depress performance — matching Scalable's gross "PnL since
            # inception" display). The broker fee, by contrast, is a real
            # cost and stays inside performance.
            tax = row.tax if pd.notna(row.tax) else 0.0
            fee = row.fee if pd.notna(row.fee) else 0.0
            cash_delta.loc[d] += (amount - tax - fee)
            if tax != 0:
                external.loc[d] -= tax
                tax_paid.loc[d] += tax
        elif ttype == "Corporate action":
            shares_delta.loc[d, isin] += shares  # can be negative (write-off)
        elif ttype == "Security transfer":
            shares_delta.loc[d, isin] += shares
            # No cash moves, but the portfolio just gained or lost
            # ``shares * price`` worth of securities. Treat that as an
            # external flow so TWR doesn't see it as performance.
            price = row.price if pd.notna(row.price) else 0.0
            if price > 0 and shares != 0:
                external.loc[d] += shares * price

    holdings = shares_delta.cumsum()
    cash = cash_delta.cumsum()

    # Align prices to the daily index; forward-fill weekends/holidays.
    px = prices.reindex(dates).ffill()
    for c in isins:
        if c not in px.columns:
            px[c] = np.nan
    px = px[isins].ffill().bfill()

    # Override today's prices with live mid-quotes when available.
    if live_prices:
        today = dates[-1]
        for isin, mid in live_prices.items():
            if isin in px.columns and mid and not np.isnan(mid):
                px.loc[today, isin] = mid

    # On days where prices are missing for a still-held position, forward-fill
    # again to bridge any remaining gaps.
    px = px.ffill().bfill()

    holdings_value = (holdings.values * px.values).sum(axis=1)
    value = pd.Series(holdings_value, index=dates) + cash

    return ValuePanel(dates=dates, holdings=holdings, cash=cash,
                      prices=px, value=value, external_flow=external,
                      tax_paid=tax_paid)


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------

def window_start(reference: pd.Timestamp, period: str,
                 absolute_start: pd.Timestamp | None = None) -> pd.Timestamp:
    """Resolve a label like ``YTD`` / ``1M`` / ``MAX`` to a starting date."""
    period = period.upper()
    if period == "MAX":
        if absolute_start is None:
            raise ValueError("MAX requires absolute_start")
        return absolute_start
    if period == "YTD":
        # Scalable's app anchors YTD at the last trading day of the previous
        # year (typically 30 Dec) rather than 1 Jan, so we do the same.
        return pd.Timestamp(reference.year - 1, 12, 30)
    deltas = {
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),
        "3M": timedelta(days=91),
        "6M": timedelta(days=182),
        "1Y": timedelta(days=365),
        "3Y": timedelta(days=3 * 365),
        "5Y": timedelta(days=5 * 365),
    }
    if period not in deltas:
        raise ValueError(f"Unknown period: {period}")
    return reference - deltas[period]


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------

def simple_return(value: pd.Series, external_flow: pd.Series) -> float:
    """Money-on-money return over a window:
    ``(V_end - V_start - net_flow) / (V_start + max(net_flow, 0))``.

    This treats deposits as additional invested capital but doesn't penalise
    the window for in/out churn (i.e. matching deposit-then-withdrawal of the
    same amount is a no-op, not a doubled denominator).
    """
    if value.empty:
        return np.nan
    v0 = value.iloc[0]
    v1 = value.iloc[-1]
    net_flow = external_flow.iloc[1:].sum()
    denom = v0 + max(net_flow, 0.0)
    if denom <= 0:
        return np.nan
    return (v1 - v0 - net_flow) / denom


def twr_series(value: pd.Series, external_flow: pd.Series) -> pd.Series:
    """Daily-chained Time-Weighted Return index starting at 1.0.

    For each day we treat external flows as occurring at the start of the day,
    so the period return is ``v_end / (v_start_prev + flow)``.
    """
    if value.empty:
        return value
    idx = value.index
    twr_index = np.zeros(len(idx))
    twr_index[0] = 1.0
    for i in range(1, len(idx)):
        v_prev = value.iloc[i - 1]
        v_cur = value.iloc[i]
        flow = external_flow.iloc[i]
        denom = v_prev + flow
        if denom <= 0 or np.isnan(v_prev) or np.isnan(v_cur):
            r = 0.0
        else:
            r = v_cur / denom - 1
        twr_index[i] = twr_index[i - 1] * (1 + r)
    return pd.Series(twr_index, index=idx)


def twr_return(value: pd.Series, external_flow: pd.Series) -> float:
    s = twr_series(value, external_flow)
    if s.empty:
        return np.nan
    return s.iloc[-1] - 1


def annualised_return(total_return: float, days: int) -> float:
    if days <= 0 or np.isnan(total_return) or total_return <= -1:
        return np.nan
    return (1 + total_return) ** (365.25 / days) - 1


def max_drawdown(value: pd.Series) -> tuple[float, pd.Timestamp | None,
                                            pd.Timestamp | None]:
    if value.empty:
        return np.nan, None, None
    running_max = value.cummax()
    dd = value / running_max - 1
    trough = dd.idxmin()
    if pd.isna(trough):
        return np.nan, None, None
    peak = value.loc[:trough].idxmax()
    return float(dd.loc[trough]), peak, trough


def volatility(twr_idx: pd.Series, periods_per_year: int = 252) -> float:
    if len(twr_idx) < 3:
        return np.nan
    daily = twr_idx.pct_change().dropna()
    if daily.empty:
        return np.nan
    return float(daily.std() * np.sqrt(periods_per_year))


def sharpe(twr_idx: pd.Series, risk_free: float = 0.02,
           periods_per_year: int = 252) -> float:
    if len(twr_idx) < 3:
        return np.nan
    daily = twr_idx.pct_change().dropna()
    if daily.empty:
        return np.nan
    excess = daily - risk_free / periods_per_year
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino(twr_idx: pd.Series, risk_free: float = 0.02,
            periods_per_year: int = 252) -> float:
    if len(twr_idx) < 3:
        return np.nan
    daily = twr_idx.pct_change().dropna()
    downside = daily[daily < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    excess = daily - risk_free / periods_per_year
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> float:
    """Annualised IRR for irregular cash flows. Sign convention: deposits
    (money going into the portfolio) are negative; the final portfolio value
    is positive.
    """
    if len(cash_flows) < 2:
        return np.nan
    t0 = cash_flows[0][0]
    times = np.array([(t - t0).days / 365.25 for t, _ in cash_flows])
    amts = np.array([c for _, c in cash_flows], dtype=float)
    if (amts > 0).sum() == 0 or (amts < 0).sum() == 0:
        return np.nan

    def npv(r: float) -> float:
        return float(np.sum(amts / (1 + r) ** times))

    try:
        return brentq(npv, -0.999, 10.0, maxiter=200, xtol=1e-6)
    except ValueError:
        return np.nan


def compute_metrics(value: pd.Series, external_flow: pd.Series,
                    risk_free: float = 0.02) -> dict:
    """Aggregate a standard set of metrics over a value window."""
    if value.empty:
        return {}
    days = (value.index[-1] - value.index[0]).days or 1
    s_ret = simple_return(value, external_flow)
    t_idx = twr_series(value, external_flow)
    t_ret = t_idx.iloc[-1] - 1 if not t_idx.empty else np.nan
    ann = annualised_return(t_ret, days) if not np.isnan(t_ret) else np.nan
    mdd, peak, trough = max_drawdown(value)
    vol = volatility(t_idx)
    shr = sharpe(t_idx, risk_free)
    srt = sortino(t_idx, risk_free)
    abs_pnl = float(value.iloc[-1] - value.iloc[0] - external_flow.iloc[1:].sum())
    return {
        "start": value.index[0], "end": value.index[-1], "days": days,
        "start_value": float(value.iloc[0]),
        "end_value": float(value.iloc[-1]),
        "net_external_flow": float(external_flow.iloc[1:].sum()),
        "absolute_pnl": abs_pnl,
        "simple_return": s_ret,
        "twr": t_ret,
        "annualised_twr": ann,
        "max_drawdown": mdd,
        "drawdown_peak": peak,
        "drawdown_trough": trough,
        "volatility": vol,
        "sharpe": shr,
        "sortino": srt,
    }
