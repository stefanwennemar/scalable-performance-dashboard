"""Derivatives sleeve: turbos, warrants and factor certificates on their own.

The sleeve holds no cash of its own — every trade is funded from and paid
back into the main account — so a time-weighted % return is meaningless
(the sleeve sits at €0 most days). Performance is expressed in euros:

    pnl[t] = market value of derivative holdings[t] − net cash invested[t]

where a buy invests ``|amount| + fee`` and a sell / write-off returns its
proceeds net of fee (gross of tax, like the rest of the dashboard). At the
end of any window this equals realized + unrealized P&L on derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_loader import is_derivative
from .performance import ValuePanel
from .portfolio import PortfolioState


@dataclass
class DerivativeSleeve:
    isins: set[str]
    pnl: pd.Series             # daily cumulative mark-to-market P&L (EUR)
    realized_pnl: pd.Series    # daily cumulative realized P&L (EUR, step)
    tax: pd.Series             # daily tax paid on derivatives (refunds < 0)
    fees: pd.Series            # daily broker fees on derivative trades
    invested: pd.Series        # daily gross buy volume (EUR, incl. fees)
    realized: pd.DataFrame     # realized trades on derivatives


def derivative_isins(tx: pd.DataFrame) -> set[str]:
    return set(tx.loc[tx["description"].map(is_derivative), "isin"].dropna())


def build_sleeve(tx: pd.DataFrame, portfolio: PortfolioState,
                 panel: ValuePanel) -> DerivativeSleeve:
    isins = derivative_isins(tx)
    dates = panel.dates
    cols = [i for i in panel.holdings.columns if i in isins]
    holdings_value = (panel.holdings[cols] * panel.prices[cols]).sum(axis=1)

    net_invested = pd.Series(0.0, index=dates)
    invested = pd.Series(0.0, index=dates)
    tax = pd.Series(0.0, index=dates)
    fees = pd.Series(0.0, index=dates)
    sub = tx[tx["isin"].isin(isins)]
    for r in sub.itertuples(index=False):
        d = pd.Timestamp(r.datetime.date())
        if d not in net_invested.index:
            continue
        amount = r.amount if pd.notna(r.amount) else 0.0
        fee = r.fee if pd.notna(r.fee) else 0.0
        if r.assetType == "Security":
            if r.type in ("Buy", "Savings plan", "Reinvestment_Distribution"):
                net_invested.loc[d] += abs(amount) + fee
                invested.loc[d] += abs(amount) + fee
                fees.loc[d] += fee
            elif r.type == "Sell":
                net_invested.loc[d] -= amount - fee
                fees.loc[d] += fee
                tax.loc[d] += r.tax if pd.notna(r.tax) else 0.0
            elif r.type == "Security transfer" and pd.notna(r.price):
                net_invested.loc[d] += (r.shares or 0.0) * r.price
        elif r.type in ("Knock-out", "Corporate action"):
            net_invested.loc[d] -= amount          # write-off proceeds
        elif r.type == "Tax refund":
            tax.loc[d] -= amount

    pnl = holdings_value - net_invested.cumsum()

    rows = [x.__dict__ for x in portfolio.realized if x.isin in isins]
    realized = pd.DataFrame(rows)
    realized_daily = pd.Series(0.0, index=dates)
    if not realized.empty:
        by_day = (realized.assign(day=realized["sell_datetime"].dt.normalize())
                  .groupby("day")["realized_pnl"].sum())
        realized_daily = by_day.reindex(dates, fill_value=0.0)
    return DerivativeSleeve(
        isins=isins, pnl=pnl, realized_pnl=realized_daily.cumsum(),
        tax=tax, fees=fees, invested=invested, realized=realized,
    )
