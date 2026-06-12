"""Historical daily prices for each ISIN.

We try ISIN -> Yahoo ticker via yfinance's symbol search, cache aggressively
on disk, and fall back to a transaction-price interpolation when Yahoo has
nothing useful (rare turbo/short certificates etc.).

The output is always denominated in EUR. For non-EUR tickers we convert using
the daily FX rate from Yahoo (``XXXEUR=X``).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from .performance import effective_today

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
SYMBOL_CACHE = os.path.join(CACHE_DIR, "isin_to_yahoo.json")
PRICE_CACHE_DIR = os.path.join(CACHE_DIR, "prices")
os.makedirs(PRICE_CACHE_DIR, exist_ok=True)

PRICE_CACHE_TTL_SECONDS = 12 * 60 * 60  # refresh daily prices twice a day
YAHOO_THROTTLE_SECONDS = 0.8            # rough rate-limit budget


_last_yahoo_call: list[float] = [0.0]


def _throttle():
    """Sleep just enough to avoid Yahoo's anonymous-search rate limit."""
    now = time.time()
    wait = YAHOO_THROTTLE_SECONDS - (now - _last_yahoo_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_yahoo_call[0] = time.time()


def _load_symbol_cache() -> dict[str, str | None]:
    if os.path.exists(SYMBOL_CACHE):
        try:
            with open(SYMBOL_CACHE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_symbol_cache(cache: dict[str, str | None]) -> None:
    tmp = SYMBOL_CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, SYMBOL_CACHE)


# Exchanges that quote in EUR — we strongly prefer them so we don't need FX.
_EUR_EXCHANGES = {"GER", "FRA", "MUN", "STU", "DUS", "BER", "HAM",  # German venues
                  "AMS", "BRU", "EBS", "PAR", "MIL", "MCE", "LIS", "HEL", "VIE"}


def _rank_quote(q: dict) -> tuple:
    exch = (q.get("exchange") or "").upper()
    qtype = q.get("quoteType")
    return (
        0 if exch in _EUR_EXCHANGES else 1,
        0 if qtype in ("ETF", "EQUITY", "MUTUALFUND") else 1,
        # XETRA / Frankfurt listings tend to have the best EUR liquidity.
        0 if exch in ("GER", "FRA") else 1,
    )


def isin_to_yahoo_symbol(isin: str,
                         cache: dict[str, str | None] | None = None) -> str | None:
    """Resolve ``isin`` to a Yahoo Finance symbol, preferring EUR-quoted
    European listings. Uses ``yfinance.Search`` (handles Yahoo auth/cookies)
    and caches results on disk."""
    cache = cache if cache is not None else _load_symbol_cache()
    if isin in cache:
        return cache[isin]
    _throttle()
    try:
        results = yf.Search(isin, max_results=10).quotes
    except Exception:
        results = []
    sym: str | None = None
    if results:
        results.sort(key=_rank_quote)
        for q in results:
            s = q.get("symbol")
            if s:
                sym = s
                break
    cache[isin] = sym
    _save_symbol_cache(cache)
    return sym


def _price_cache_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(PRICE_CACHE_DIR, f"{safe}.pkl")


def _load_cached_prices(symbol: str) -> pd.DataFrame | None:
    p = _price_cache_path(symbol)
    if not os.path.exists(p):
        return None
    age = time.time() - os.path.getmtime(p)
    if age > PRICE_CACHE_TTL_SECONDS:
        return None
    try:
        return pd.read_pickle(p)
    except (OSError, ValueError):
        return None


def _save_cached_prices(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_pickle(_price_cache_path(symbol))


def _download_history(symbol: str, start: datetime) -> pd.DataFrame:
    """Download adjusted daily closes from Yahoo. Returns columns:
    ``close`` (in symbol's currency) and ``currency``."""
    _throttle()
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(start=start.strftime("%Y-%m-%d"),
                          interval="1d", auto_adjust=True,
                          actions=False, raise_errors=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        # Some intervals come back tz-aware -> normalise to naive dates.
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        info_cur = None
        try:
            info_cur = tk.fast_info.get("currency")
        except Exception:
            pass
        out = pd.DataFrame({"close": hist["Close"].astype(float)})
        out["currency"] = (info_cur or "EUR").upper()
        # London listings often quote in GBp (pence). Convert to GBP.
        if symbol.endswith(".L") and out["currency"].iloc[-1] in ("GBP", "GBp"):
            out["close"] = out["close"] / 100.0
            out["currency"] = "GBP"
        return out
    except Exception:
        return pd.DataFrame()


def _fx_to_eur(currency: str, start: datetime,
               fx_cache: dict[str, pd.Series]) -> pd.Series:
    """Daily multiplier to convert ``currency`` to EUR (i.e. ``EUR = local * fx``)."""
    if currency == "EUR":
        # Trivial unit series.
        idx = pd.date_range(start, effective_today().date(), freq="D")
        return pd.Series(1.0, index=idx)
    if currency in fx_cache:
        return fx_cache[currency]
    sym = f"{currency}EUR=X"
    df = _load_cached_prices(sym)
    if df is None or df.empty:
        df = _download_history(sym, start - timedelta(days=10))
        if not df.empty:
            _save_cached_prices(sym, df)
    if df.empty:
        idx = pd.date_range(start, effective_today().date(), freq="D")
        fx_cache[currency] = pd.Series(np.nan, index=idx)
    else:
        fx_cache[currency] = df["close"]
    return fx_cache[currency]


def get_price_series(isin: str, start: datetime,
                     fallback_prices: pd.DataFrame | None = None,
                     symbol_cache: dict[str, str | None] | None = None,
                     fx_cache: dict[str, pd.Series] | None = None,
                     try_yahoo: bool = True,
                     ) -> pd.Series:
    """Return daily EUR close prices for ``isin`` from ``start`` onwards.

    ``fallback_prices`` is the per-ISIN slice of the transaction history
    (datetime, price) — used to interpolate when Yahoo has no data.

    Pass ``try_yahoo=False`` for closed positions where we only need
    approximate prices for the historical value chart — this skips the slow
    Yahoo symbol lookup and goes straight to transaction-price interpolation.
    """
    fx_cache = fx_cache if fx_cache is not None else {}
    symbol_cache = (symbol_cache if symbol_cache is not None
                    else _load_symbol_cache())

    df: pd.DataFrame | None = None
    if try_yahoo:
        symbol = isin_to_yahoo_symbol(isin, cache=symbol_cache)
        if symbol:
            df = _load_cached_prices(symbol)
            if df is None:
                df = _download_history(symbol, start - timedelta(days=5))
                if not df.empty:
                    _save_cached_prices(symbol, df)

    today = effective_today()
    daily_idx = pd.date_range(start.date(), today, freq="D")

    if df is not None and not df.empty:
        cur = df["currency"].iloc[-1] if "currency" in df.columns else "EUR"
        series_local = df["close"].copy()
        series_local = series_local.reindex(
            series_local.index.union(daily_idx)).sort_index().ffill().reindex(daily_idx)
        if cur != "EUR":
            fx = _fx_to_eur(cur, start, fx_cache)
            fx_aligned = fx.reindex(daily_idx).ffill().bfill()
            series_eur = series_local * fx_aligned
        else:
            series_eur = series_local
        if series_eur.notna().any():
            return series_eur.ffill().bfill()

    # Yahoo had nothing useful — interpolate from transaction prices.
    if fallback_prices is not None and not fallback_prices.empty:
        fp = fallback_prices.dropna(subset=["price"]).copy()
        fp = fp[fp["price"] > 0]
        if not fp.empty:
            fp["date"] = pd.to_datetime(fp["datetime"]).dt.normalize()
            fp = fp.drop_duplicates(subset="date", keep="last")
            s = pd.Series(fp["price"].values, index=fp["date"].values)
            s = s.sort_index()
            s = s.reindex(s.index.union(daily_idx)).sort_index()
            s = s.interpolate("time").ffill().bfill().reindex(daily_idx)
            return s

    return pd.Series(np.nan, index=daily_idx)


def get_all_prices(isins: list[str], transactions: pd.DataFrame,
                   start: datetime,
                   yahoo_isins: set[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame of daily EUR prices, columns = ISINs.

    Only ISINs in ``yahoo_isins`` go through Yahoo's symbol search /
    history download. The rest get transaction-price interpolation, which is
    accurate enough for the historical value chart of long-closed positions.
    If ``yahoo_isins`` is None, all ISINs are tried via Yahoo.
    """
    symbol_cache = _load_symbol_cache()
    fx_cache: dict[str, pd.Series] = {}
    out = {}
    for isin in isins:
        tx_slice = transactions[transactions["isin"] == isin][["datetime", "price"]]
        try_yahoo = (yahoo_isins is None) or (isin in yahoo_isins)
        out[isin] = get_price_series(isin, start, fallback_prices=tx_slice,
                                     symbol_cache=symbol_cache,
                                     fx_cache=fx_cache,
                                     try_yahoo=try_yahoo)
    return pd.DataFrame(out)
