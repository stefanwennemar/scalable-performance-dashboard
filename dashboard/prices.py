"""Live price fetcher for gettex.de.

The gettex.de detail page renders bid/ask in the document title once a
client-side SAML handshake + LSEG streaming subscription completes. There is
no clean REST endpoint we can hit anonymously, so we use a headless Chromium
browser via Playwright to wait for the title to populate.

We probe the three possible URL slugs (``/fond/``, ``/aktie/``, ``/zertifikat/``)
in parallel for the same ISIN inside one browser context and take the first
that returns a real numeric title. Results are cached on disk to keep the
dashboard snappy and avoid hammering the gettex servers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "live_prices.json")
CACHE_TTL_SECONDS = 15 * 60       # 15 min — refresh button is one click away
STALE_TTL_SECONDS = 24 * 60 * 60  # keep entries up to a day for fallback

URL_SLUGS = ("fond", "aktie", "zertifikat", "anleihe", "etc")
GETTEX_BASE = "https://www.gettex.de"

_BID_ASK_RE = re.compile(r"^\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*\|")

# German certificates / Turbos / Optionsscheine have ISINs starting with
# DE000 + an issuer letter (G = Goldman Sachs, H = HSBC, V = Vontobel,
# X = Citi, etc). For those we know the right page slug a priori, so we
# bias the slug probe order to start with ``zertifikat``.
_CERT_ISSUER_LETTERS = set("GHVXSCB")

# Names that come back when gettex serves a "wrong" or default page —
# treating them as a successful scrape would poison the slug cache, which
# is exactly the bug that left Apple cached as ``zertifikat``.
_SUSPICIOUS_NAME_TOKENS = (
    "gettex", "willkommen", "fehler", "error", "{name}", "{wkn}",
)


def slug_hints_from_isins(isins: Iterable[str]) -> dict[str, str]:
    """Best-effort a-priori slug hint based on ISIN prefix conventions.

    The cached slug (if any) still takes precedence inside ``_fetch_one``;
    this is the hint we use the *first* time we see an ISIN.
    """
    hints: dict[str, str] = {}
    for isin in isins:
        if (len(isin) >= 6 and isin[:5] == "DE000"
                and isin[5] in _CERT_ISSUER_LETTERS):
            hints[isin] = "zertifikat"
    return hints


def _name_looks_real(name: str | None, isin: str) -> bool:
    """Sanity check on the security name parsed out of the page title.

    We've seen gettex occasionally serve a generic page whose title still
    has the BID|ASK shape (because of a fallback widget), making the
    scraper think it succeeded. The cure: reject anything where the name
    field is empty, suspiciously short, or contains one of the markers
    that show up on the homepage / error pages.
    """
    if not name:
        return False
    n = name.strip().lower()
    if len(n) < 3:
        return False
    for tok in _SUSPICIOUS_NAME_TOKENS:
        if tok in n:
            return False
    return True


@dataclass
class LivePrice:
    isin: str
    bid: float | None
    ask: float | None
    mid: float | None
    name: str | None
    wkn: str | None
    slug: str | None        # which gettex section worked (fond/aktie/...)
    fetched_at: float       # unix timestamp
    source: str = "gettex"  # may be "cache" or "yahoo" for fallbacks

    @property
    def fresh(self) -> bool:
        return (time.time() - self.fetched_at) < CACHE_TTL_SECONDS


def _parse_german_number(s: str) -> float | None:
    s = s.strip().replace("\xa0", "")
    if not s or s == "-":
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_title(title: str) -> tuple[float | None, float | None, str | None, str | None]:
    """Extract (bid, ask, name, wkn) from the gettex page title.

    Real titles look like ``42,57 | 42,59 | Vanguard USD ... | A12HJF``.
    """
    if not title or "{BID}" in title:
        return None, None, None, None
    parts = [p.strip() for p in title.split("|")]
    if len(parts) < 2:
        return None, None, None, None
    bid = _parse_german_number(parts[0])
    ask = _parse_german_number(parts[1])
    # gettex prints "-" or "0,00" when one side is missing; treat <= 0 as None.
    if bid is not None and bid <= 0:
        bid = None
    if ask is not None and ask <= 0:
        ask = None
    name = parts[2] if len(parts) > 2 else None
    wkn = parts[3] if len(parts) > 3 else None
    return bid, ask, name, wkn


def _load_cache() -> dict[str, dict]:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_FILE)


async def _fetch_one(context, isin: str, preferred_slug: str | None,
                     timeout_ms: int = 15000) -> LivePrice:
    """Try each gettex slug until the page title resolves to a real quote."""
    slugs = []
    if preferred_slug and preferred_slug in URL_SLUGS:
        slugs.append(preferred_slug)
    slugs.extend(s for s in URL_SLUGS if s not in slugs)

    last_err: Exception | None = None
    for slug in slugs:
        url = f"{GETTEX_BASE}/{slug}/{isin}/"
        page = await context.new_page()
        try:
            resp = await page.goto(url, wait_until="domcontentloaded",
                                   timeout=timeout_ms)
            if resp is None or resp.status >= 400:
                continue
            try:
                # Title is set by JS only after SAML+streaming completes.
                await page.wait_for_function(
                    "document.title && !document.title.startsWith('{BID}') "
                    "&& /^\\s*[\\d.,]+\\s*\\|/.test(document.title)",
                    timeout=timeout_ms,
                )
            except PWTimeout:
                continue
            title = await page.title()
            bid, ask, name, wkn = _parse_title(title)
            if bid is None and ask is None:
                continue
            # Guard against gettex serving a default / error page that
            # happens to satisfy the BID|ASK regex. If the name doesn't
            # look like a real security name, treat as a miss and keep
            # iterating slugs.
            if not _name_looks_real(name, isin):
                continue
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2
            else:
                mid = bid if bid is not None else ask
            return LivePrice(
                isin=isin, bid=bid, ask=ask, mid=mid,
                name=name, wkn=wkn, slug=slug,
                fetched_at=time.time(), source="gettex",
            )
        except Exception as e:
            last_err = e
        finally:
            await page.close()

    return LivePrice(isin=isin, bid=None, ask=None, mid=None,
                     name=None, wkn=None, slug=None,
                     fetched_at=time.time(), source="error")


async def _fetch_many(isins: list[str], slug_hints: dict[str, str],
                      concurrency: int = 4) -> dict[str, LivePrice]:
    results: dict[str, LivePrice] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
            viewport={"width": 1280, "height": 800},
            locale="de-DE",
        )
        # Block heavy non-essential resources to speed things up.
        async def _route(route):
            r = route.request
            if r.resource_type in ("image", "media", "font"):
                await route.abort()
            else:
                await route.continue_()
        await context.route("**/*", _route)

        sem = asyncio.Semaphore(concurrency)

        async def worker(isin: str):
            async with sem:
                results[isin] = await _fetch_one(
                    context, isin, slug_hints.get(isin))

        await asyncio.gather(*(worker(i) for i in isins))
        await context.close()
        await browser.close()
    return results


def fetch_live_prices(isins: Iterable[str],
                      slug_hints: dict[str, str] | None = None,
                      force_refresh: bool = False,
                      concurrency: int = 4) -> dict[str, LivePrice]:
    """Return a {isin: LivePrice} mapping, using disk cache for fresh entries."""
    isins = list(dict.fromkeys(isins))   # de-dupe, preserve order
    cache = _load_cache()
    now = time.time()
    out: dict[str, LivePrice] = {}
    to_fetch: list[str] = []

    for isin in isins:
        entry = cache.get(isin)
        if entry and not force_refresh:
            age = now - entry.get("fetched_at", 0)
            if age < CACHE_TTL_SECONDS and entry.get("bid") is not None:
                lp = LivePrice(**entry)
                lp.source = "cache"
                out[isin] = lp
                continue
        to_fetch.append(isin)

    if to_fetch:
        # Playwright must run in its own thread to avoid clashing with the
        # Dash/Flask event loop already on the main thread.
        slug_hints = slug_hints or {}

        def runner():
            return asyncio.run(_fetch_many(
                to_fetch, slug_hints, concurrency=concurrency))

        with ThreadPoolExecutor(max_workers=1) as ex:
            fresh = ex.submit(runner).result()

        for isin, lp in fresh.items():
            out[isin] = lp
            if lp.bid is not None:
                cache[isin] = asdict(lp)
            elif isin in cache and (now - cache[isin].get("fetched_at", 0)) < STALE_TTL_SECONDS:
                # Keep stale cache as fallback.
                stale = LivePrice(**cache[isin])
                stale.source = "stale-cache"
                out[isin] = stale

        _save_cache(cache)

    return out


def get_slug_hints_from_cache(isins: Iterable[str]) -> dict[str, str]:
    """Use the on-disk cache to remember which slug worked previously, so we
    don't always have to probe ETF→stock→certificate for every ISIN."""
    cache = _load_cache()
    hints: dict[str, str] = {}
    for isin in isins:
        entry = cache.get(isin)
        if entry and entry.get("slug"):
            hints[isin] = entry["slug"]
    return hints


def datetime_of(lp: LivePrice) -> datetime:
    return datetime.fromtimestamp(lp.fetched_at)
