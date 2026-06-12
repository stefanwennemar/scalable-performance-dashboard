"""Historical daily closes from gettex.de (Munich exchange).

gettex's chart widget calls a SAML-authenticated REST endpoint at
``lseg-widgets.financial.com``. We piggyback on the same browser session that
the live-price scraper already uses: open one page so dory bootstraps the
SAML handshake, then run ``dory.Rest.get(...)`` calls inside the page via
``page.evaluate``.

Returned prices are in EUR and align exactly with what the Scalable broker
quotes — no FX conversion needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
from playwright.async_api import async_playwright

from .performance import effective_today

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
RIC_CACHE = os.path.join(CACHE_DIR, "isin_to_ric.json")
GETTEX_PRICE_DIR = os.path.join(CACHE_DIR, "gettex_prices")
os.makedirs(GETTEX_PRICE_DIR, exist_ok=True)
PRICE_TTL_SECONDS = 12 * 60 * 60

BOOTSTRAP_URL = "https://www.gettex.de/fond/IE00BDD48R20/"


def _load_ric_cache() -> dict[str, str | None]:
    if os.path.exists(RIC_CACHE):
        try:
            with open(RIC_CACHE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_ric_cache(cache: dict[str, str | None]) -> None:
    tmp = RIC_CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, RIC_CACHE)


def _price_cache_path(isin: str) -> str:
    return os.path.join(GETTEX_PRICE_DIR, f"{isin}.pkl")


def _load_cached_prices(isin: str) -> pd.DataFrame | None:
    p = _price_cache_path(isin)
    if not os.path.exists(p):
        return None
    if (time.time() - os.path.getmtime(p)) > PRICE_TTL_SECONDS:
        return None
    try:
        return pd.read_pickle(p)
    except (OSError, ValueError):
        return None


def _save_cached_prices(isin: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_pickle(_price_cache_path(isin))


# JS helpers we run inside the page context. The page already has `dory`
# loaded after networkidle, so these calls reuse the SAML session.
_JS_FIND_RIC = """
async (isin) => {
    const r = await dory.Rest.get(
        '/find/securities'
        + '?fids=q.RIC,x._ISIN'
        + '&search=' + isin
        + '&searchFor=ISIN'
        + '&exchanges=GTX'
        + '&pageSize=2&pageNo=0');
    const rows = r && r.data && r.data.data;
    if (rows && rows.length && rows[0]['q.RIC']) return rows[0]['q.RIC'];
    return null;
}
"""

_JS_FETCH_HISTORY = """
async ({ric, fromDate, toDate}) => {
    const url = '/timeseries/historical'
        + '?ric=' + ric
        + '&fids=_DATE_END,CLOSE_PRC'
        + '&samples=D'
        + '&appendRecentData=all'
        + '&adjustment=D'
        + '&toDate=' + toDate
        + '&fromDate=' + fromDate;
    const r = await dory.Rest.get(url);
    return r && r.data && r.data.data;
}
"""


async def _fetch_async(isins: list[str], start: datetime, end: datetime,
                       ric_hints: dict[str, str | None],
                       concurrency: int = 1
                       ) -> dict[str, pd.DataFrame]:
    """Use a single browser context to fetch history for many ISINs."""
    from_str = start.strftime("%Y-%m-%dT00:00:00")
    to_str = end.strftime("%Y-%m-%dT23:59:59")
    out: dict[str, pd.DataFrame] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
            locale="de-DE",
            viewport={"width": 1280, "height": 800},
        )

        async def _route(route):
            r = route.request
            if r.resource_type in ("image", "media", "font"):
                await route.abort()
            else:
                await route.continue_()
        await ctx.route("**/*", _route)

        page = await ctx.new_page()
        try:
            await page.goto(BOOTSTRAP_URL, wait_until="networkidle",
                            timeout=30000)
            await page.wait_for_function(
                "typeof dory !== 'undefined' && dory.Rest && dory.Auth",
                timeout=15000,
            )
        except Exception as e:
            await browser.close()
            raise RuntimeError(f"gettex bootstrap failed: {e}") from e

        async def fetch_one(isin: str):
            ric = ric_hints.get(isin)
            try:
                if ric is None:
                    ric = await page.evaluate(_JS_FIND_RIC, isin)
                if not ric:
                    return isin, None, None
                rows = await page.evaluate(_JS_FETCH_HISTORY, {
                    "ric": ric, "fromDate": from_str, "toDate": to_str,
                })
                if not rows:
                    return isin, ric, None
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["_DATE_END"])
                df["close"] = pd.to_numeric(df["CLOSE_PRC"], errors="coerce")
                df = df[["date", "close"]].dropna()
                df = df.set_index("date").sort_index()
                return isin, ric, df
            except Exception:
                return isin, ric, None

        # gettex's dory.Rest has its own rate-limit handling; we keep
        # concurrency at 1 inside the page anyway.
        for isin in isins:
            isin_, ric_, df_ = await fetch_one(isin)
            ric_hints[isin_] = ric_
            if df_ is not None and not df_.empty:
                out[isin_] = df_

        await browser.close()
    return out


def fetch_gettex_history(isins: list[str], start: datetime,
                         end: datetime | None = None,
                         force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch daily EUR closes from gettex for each ISIN.

    Returns ``{isin: DataFrame[date -> close]}`` for whatever ISINs gettex
    has data for. ISINs without a gettex listing are silently skipped (the
    caller can fall back to yfinance).
    """
    isins = list(dict.fromkeys(isins))
    end = end or effective_today().to_pydatetime()
    ric_cache = _load_ric_cache()

    out: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for isin in isins:
        if not force:
            df = _load_cached_prices(isin)
            if df is not None and not df.empty:
                out[isin] = df
                continue
        if ric_cache.get(isin) is None and isin in ric_cache:
            # Cached negative lookup (no gettex listing) — skip.
            continue
        to_fetch.append(isin)

    if to_fetch:
        def runner():
            return asyncio.run(_fetch_async(to_fetch, start, end, ric_cache))

        with ThreadPoolExecutor(max_workers=1) as ex:
            fetched = ex.submit(runner).result()

        for isin in to_fetch:
            df = fetched.get(isin)
            if df is not None and not df.empty:
                _save_cached_prices(isin, df)
                out[isin] = df
            else:
                ric_cache.setdefault(isin, None)
        _save_ric_cache(ric_cache)

    return out
