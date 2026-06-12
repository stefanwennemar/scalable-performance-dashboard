"""Benchmark series and market-related metrics.

The benchmark is the iShares MSCI ACWI UCITS ETF (ISIN ``IE00B6R52259``,
WKN A1JMDF, ticker IUSQ on Xetra / gettex). Prices come from the same
gettex historical endpoint we use for the portfolio constituents, so the
benchmark is denominated in EUR and aligns naturally with everything else.

Market metrics are computed against the benchmark as if it were "the
market":
- **beta**    — slope of portfolio daily excess return regressed on
                benchmark daily excess return.
- **alpha**   — annualised Jensen's alpha:
                ``α = (R_p − R_f) − β (R_b − R_f)``.
- **R²**      — squared Pearson correlation of daily returns; how much of
                the portfolio's movement is "explained" by the benchmark.
- **tracking error** — annualised standard deviation of the daily return
                difference (portfolio minus benchmark).
- **info ratio** — annualised active return over tracking error.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .gettex_history import fetch_gettex_history

BENCHMARK_ISIN = "IE00B6R52259"
BENCHMARK_NAME = "MSCI ACWI (IUSQ)"


def fetch_benchmark_prices(start: datetime, end: datetime,
                           force: bool = False) -> pd.Series:
    """Return daily EUR closing prices for the benchmark, or an empty
    series if gettex has nothing for the ISIN."""
    res = fetch_gettex_history([BENCHMARK_ISIN], start=start, end=end,
                               force=force)
    df = res.get(BENCHMARK_ISIN)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def compute_market_metrics(portfolio_daily: pd.Series,
                           benchmark_daily: pd.Series,
                           risk_free: float = 0.02,
                           periods_per_year: int = 252) -> dict:
    """Beta / alpha / R² / tracking error / information ratio.

    ``portfolio_daily`` and ``benchmark_daily`` are daily simple returns,
    *not* price levels. Align indices, drop NaNs, and require at least 10
    points before computing.
    """
    aligned = pd.concat(
        [portfolio_daily, benchmark_daily], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return {"beta": np.nan, "alpha": np.nan, "r_squared": np.nan,
                "tracking_error": np.nan, "info_ratio": np.nan}

    p = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    rf_d = risk_free / periods_per_year

    cov_pb = float(((p - p.mean()) * (b - b.mean())).mean())
    var_b = float(((b - b.mean()) ** 2).mean())
    beta = cov_pb / var_b if var_b > 0 else np.nan

    p_ann = float(p.mean()) * periods_per_year
    b_ann = float(b.mean()) * periods_per_year
    if np.isnan(beta):
        alpha = np.nan
    else:
        alpha = (p_ann - risk_free) - beta * (b_ann - risk_free)

    corr = float(p.corr(b))
    r_squared = corr ** 2 if not np.isnan(corr) else np.nan

    diff = p - b
    tracking_error = float(diff.std()) * np.sqrt(periods_per_year)
    if tracking_error > 0:
        info_ratio = (p_ann - b_ann) / tracking_error
    else:
        info_ratio = np.nan

    return {
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
    }
