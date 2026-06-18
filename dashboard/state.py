"""Process-wide state for the dashboard: load data once, lazily refresh
prices/value-panel as needed. Kept separate from the Dash app so callbacks
can stay thin.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .benchmark import BENCHMARK_ISIN, BENCHMARK_NAME, fetch_benchmark_prices
from .data_loader import load_transactions, isin_descriptions, LoadedTransactions
from .gettex_history import fetch_gettex_history
from .historical_prices import get_all_prices
from .performance import build_value_panel, effective_today, ValuePanel
from .portfolio import (build_portfolio, positions_to_dataframe,
                        realized_to_dataframe, PortfolioState)
from .prices import (fetch_live_prices, get_slug_hints_from_cache,
                     slug_hints_from_isins,
                     LivePrice, CACHE_TTL_SECONDS)


PRICE_PANEL_TTL_SECONDS = 12 * 60 * 60


@dataclass
class DashboardState:
    tx: LoadedTransactions
    portfolio: PortfolioState
    descriptions: dict[str, str]
    positions_df: pd.DataFrame                # current positions table
    realized_df: pd.DataFrame                 # realized trades table
    live_prices: dict[str, LivePrice] = field(default_factory=dict)
    live_prices_at: float = 0.0
    panel: ValuePanel | None = None
    panel_at: float = 0.0
    benchmark: pd.Series | None = None       # daily EUR closes for IUSQ
    benchmark_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)


_state: DashboardState | None = None
_state_lock = threading.Lock()


def load_state(force: bool = False) -> DashboardState:
    """Load (or reload) all transaction-derived state. Safe to call from any
    thread; subsequent calls return the cached instance."""
    global _state
    with _state_lock:
        if _state is not None and not force:
            return _state
        tx = load_transactions()
        portfolio = build_portfolio(tx.raw)
        descriptions = isin_descriptions(tx.raw)
        positions_df = positions_to_dataframe(portfolio.positions)
        realized_df = realized_to_dataframe(portfolio.realized)
        _state = DashboardState(
            tx=tx,
            portfolio=portfolio,
            descriptions=descriptions,
            positions_df=positions_df,
            realized_df=realized_df,
        )
        return _state


# Guards against firing two playwright sessions in parallel — the second
# would just lose against the first and waste a Chromium launch.
_bg_refresh_lock = threading.Lock()
_bg_refresh_running = False


def _do_blocking_refresh(st, force: bool) -> dict[str, LivePrice]:
    """Run a Playwright price scrape and write the result into ``st``."""
    isins = list(st.portfolio.positions.keys())
    hints = slug_hints_from_isins(isins)
    hints.update(get_slug_hints_from_cache(isins))
    prices = fetch_live_prices(isins, slug_hints=hints,
                               force_refresh=force, concurrency=6)
    with st._lock:
        st.live_prices = prices
        st.live_prices_at = time.time()
    return prices


def _kick_background_refresh(st) -> None:
    """Spawn a daemon thread that re-scrapes live prices and overwrites the
    in-memory cache. No-op if a refresh is already running."""
    global _bg_refresh_running
    with _bg_refresh_lock:
        if _bg_refresh_running:
            return
        _bg_refresh_running = True

    def _runner():
        global _bg_refresh_running
        try:
            _do_blocking_refresh(st, force=False)
        except Exception as e:
            print(f"[bg refresh] error: {e}")
        finally:
            with _bg_refresh_lock:
                _bg_refresh_running = False

    threading.Thread(target=_runner, daemon=True, name="bg-price-refresh").start()


def refresh_live_prices(force: bool = False,
                        blocking: bool = True) -> dict[str, LivePrice]:
    """Fetch live gettex prices for all currently held positions.

    ``blocking=False`` returns the cached prices immediately and (if the
    cache is stale) kicks off a background refresh that will update the
    cache for the next call. This is used on page-load / interval-fired
    refreshes so the user never waits 2-3 minutes for the UI to render.

    ``blocking=True`` always returns fresh prices (running the scrape on
    the calling thread). Used when the user explicitly clicks the
    "Refresh prices" button — they've asked for fresh data so we deliver.
    """
    st = load_state()
    cache_fresh = (st.live_prices and
                   (time.time() - st.live_prices_at) < CACHE_TTL_SECONDS)
    if not force and cache_fresh:
        return st.live_prices
    if not blocking and st.live_prices:
        # Stale-while-revalidate: hand the caller what we have, refresh in
        # the background so the next callback gets fresh data.
        _kick_background_refresh(st)
        return st.live_prices
    # Either user explicitly forced, or there's no cache at all to fall
    # back to (first call after process start). Block on the scrape.
    return _do_blocking_refresh(st, force=force)


def get_value_panel(refresh: bool = False) -> ValuePanel:
    """Lazy-build the day-by-day portfolio value panel."""
    st = load_state()
    with st._lock:
        if (st.panel is not None and not refresh
                and (time.time() - st.panel_at) < PRICE_PANEL_TTL_SECONDS):
            return st.panel
        all_isins = sorted({i for i in st.tx.securities["isin"].dropna().unique()})
        start = pd.Timestamp(st.tx.raw["datetime"].min().date())
        open_isins = set(st.portfolio.positions.keys())

        # 1) Try gettex first for every open position (EUR-native, matches
        #    Scalable's app).
        gettex_prices = fetch_gettex_history(
            list(open_isins), start=start.to_pydatetime(),
            end=effective_today().to_pydatetime(),
        )
        gettex_isins = set(gettex_prices.keys())

        # 2) Yahoo for any open ISIN that gettex didn't return + closed
        #    positions fall through to transaction-price interpolation.
        yahoo_isins = open_isins - gettex_isins
        prices = get_all_prices(all_isins, st.tx.securities, start,
                                yahoo_isins=yahoo_isins)

        # 3) Overlay gettex data on top, taking priority where we have it.
        for isin, df in gettex_prices.items():
            if isin not in prices.columns:
                prices[isin] = float("nan")
            series = df["close"].copy()
            series.index = pd.to_datetime(series.index).normalize()
            prices[isin] = series.reindex(prices.index).ffill().bfill()

        st.panel = build_value_panel(st.tx.raw, prices)
        st.panel_at = time.time()
        return st.panel


def get_benchmark_series(force: bool = False) -> pd.Series:
    """Daily benchmark prices, reindexed onto the panel's date grid so it
    can be plotted alongside portfolio value with no further alignment."""
    st = load_state()
    panel = get_value_panel()
    with st._lock:
        if (st.benchmark is not None and not force
                and (time.time() - st.benchmark_at) < PRICE_PANEL_TTL_SECONDS):
            return st.benchmark
        start = pd.Timestamp(panel.dates[0]).to_pydatetime()
        end = pd.Timestamp(panel.dates[-1]).to_pydatetime()
        s = fetch_benchmark_prices(start, end, force=force)
        if s.empty:
            st.benchmark = pd.Series(np.nan, index=panel.dates)
        else:
            st.benchmark = s.reindex(panel.dates).ffill().bfill()
        st.benchmark_at = time.time()
        return st.benchmark


def current_value() -> float:
    """Best estimate of today's portfolio value using live prices + cash.

    Uses the same fallback chain the positions table uses so that the
    "Portfolio value" KPI matches the sum of per-position values:
      live gettex mid → panel's today close (forward-filled if needed) →
      lot avg cost. The avg-cost fallback only kicks in for positions that
      have neither a live quote nor any historical price (very rare).
    """
    st = load_state()
    refresh_live_prices()
    panel = get_value_panel()
    today = panel.dates[-1]
    px = panel.prices
    total = st.portfolio.cash_balance
    for isin, pos in st.portfolio.positions.items():
        lp = st.live_prices.get(isin)
        mid = lp.mid if (lp and lp.mid is not None) else None
        if mid is None and isin in px.columns:
            today_px = float(px.loc[today, isin])
            if today_px == today_px:    # not NaN
                mid = today_px
        if mid is None:
            mid = pos.avg_cost
        total += pos.shares * mid
    return total
