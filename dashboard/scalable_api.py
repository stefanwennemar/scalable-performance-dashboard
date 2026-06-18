"""Optional Scalable Capital broker data via the ``sc`` CLI.

If the user has installed Scalable's official CLI (https://github.com/
ScalableCapital/scalable-cli) and run ``sc login``, the dashboard prefers
its data over our gettex-scraped values. The CLI's REST calls are an
order of magnitude faster than 60 Playwright page-loads, give us live
mids for instruments gettex can't quote (e.g. open-ended turbos), and
return Scalable's own portfolio valuation and time-window performance
numbers — eliminating the structural gettex-vs-Scalable reconciliation
gap.

All access is **read-only**; we never call any of the ``broker.trade.*``
subcommands. Every public function silently falls back to ``None`` /
empty data when ``sc`` isn't present, isn't logged in, or the call
fails — so the rest of the dashboard never has to care whether the API
is connected. The original CSV-only behaviour stays intact.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

# How long to keep an API response around before re-querying. The CLI is
# fast (~150 ms) but doing it on every callback would still hammer the
# Scalable backend; 30 s feels right for a snapshot dashboard.
CACHE_TTL_SECONDS = 30

# Hard ceiling on any subprocess call so a hung CLI can't freeze a callback.
SUBPROCESS_TIMEOUT = 8


@dataclass
class ApiStatus:
    state: str          # "missing" | "logged_out" | "connected" | "error"
    name: str | None
    account_id: str | None
    detail: str | None  # human-readable error / hint, when relevant


_status: ApiStatus | None = None
_status_at: float = 0.0
_status_lock = threading.Lock()

_overview_cache: tuple[float, dict] | None = None
_holdings_cache: tuple[float, dict] | None = None
_payload_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Low-level subprocess helpers
# ---------------------------------------------------------------------------

def _sc_path() -> str | None:
    """Resolve the ``sc`` binary if it's on PATH."""
    return shutil.which("sc")


def _run_sc(args: list[str], timeout: float = SUBPROCESS_TIMEOUT
             ) -> tuple[int, str, str]:
    """Run an ``sc`` subcommand. Returns (returncode, stdout, stderr).

    Never raises; on missing CLI or timeout the returncode is non-zero
    and the caller decides what to do.
    """
    binary = _sc_path()
    if binary is None:
        return 127, "", "sc binary not found on PATH"
    try:
        res = subprocess.run(
            [binary, *args],
            capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"sc {' '.join(args)} timed out after {timeout}s"
    except (OSError, FileNotFoundError) as e:
        return 126, "", f"failed to spawn sc: {e}"


def _run_sc_json(args: list[str]) -> dict | None:
    """Run ``sc … --json`` and parse the envelope. Returns the ``data``
    payload on success, ``None`` on any failure."""
    rc, out, err = _run_sc(args)
    if rc != 0 or not out.strip():
        return None
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError:
        return None
    if not envelope.get("ok"):
        return None
    return envelope.get("data")


# ---------------------------------------------------------------------------
# Status: are we connected?
# ---------------------------------------------------------------------------

def status(force: bool = False) -> ApiStatus:
    """Inspect connection state. Cached for 30 s so the UI can poll cheaply.

    State machine:
    - ``missing``     : ``sc`` not on PATH at all
    - ``logged_out``  : ``sc`` exists but ``whoami`` doesn't return a session
    - ``connected``   : ``whoami`` returns identity
    - ``error``       : ``sc`` failed unexpectedly (network etc.)
    """
    global _status, _status_at
    with _status_lock:
        if (not force and _status is not None
                and (time.time() - _status_at) < CACHE_TTL_SECONDS):
            return _status
        binary = _sc_path()
        if binary is None:
            _status = ApiStatus(
                state="missing", name=None, account_id=None,
                detail=("Scalable CLI not installed. See "
                        "https://github.com/ScalableCapital/scalable-cli"),
            )
            _status_at = time.time()
            return _status
        rc, out, err = _run_sc(["whoami"], timeout=5)
        if rc != 0:
            # No persisted session — most common case for a brand-new user.
            looks_loggedout = ("no active session" in (out + err).lower()
                               or "not logged in" in (out + err).lower()
                               or "session" in (out + err).lower())
            _status = ApiStatus(
                state="logged_out" if looks_loggedout else "error",
                name=None, account_id=None,
                detail=(err.strip() or out.strip() or
                        "sc whoami did not return a session"),
            )
            _status_at = time.time()
            return _status
        # ``whoami`` prints YAML-ish key:value lines, not JSON. Cheap parse.
        name = account_id = None
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "name":
                name = v
            elif k == "id":
                account_id = v
        _status = ApiStatus(state="connected", name=name,
                             account_id=account_id, detail=None)
        _status_at = time.time()
        return _status


def is_available() -> bool:
    """Convenience: True iff we can call read-only endpoints right now."""
    return status().state == "connected"


# ---------------------------------------------------------------------------
# Cached payload fetchers
# ---------------------------------------------------------------------------

def _cached_fetch(args: list[str], cache_attr: str) -> dict | None:
    """Helper that wires ``_overview_cache`` / ``_holdings_cache``."""
    global _overview_cache, _holdings_cache
    with _payload_lock:
        cache = (_overview_cache if cache_attr == "overview"
                 else _holdings_cache)
        if cache is not None and (time.time() - cache[0]) < CACHE_TTL_SECONDS:
            return cache[1]
    if not is_available():
        return None
    data = _run_sc_json(args)
    if data is None:
        return None
    with _payload_lock:
        if cache_attr == "overview":
            _overview_cache = (time.time(), data)
        else:
            _holdings_cache = (time.time(), data)
    return data


def overview(include_ytd: bool = True) -> dict | None:
    """Portfolio valuation + Scalable's own performance vector. Returns
    ``data.result`` dict from ``sc broker overview --json``."""
    args = ["broker", "overview", "--json"]
    if include_ytd:
        args.append("--include-year-to-date")
    data = _cached_fetch(args, "overview")
    if data is None:
        return None
    return data.get("result") or data


def holdings() -> list[dict] | None:
    """Per-position holdings list with live mid prices. Returns a list of
    item dicts (each with ``isin``, ``quantity``, ``quote_mid_price``,
    ``fifo_price``, ``valuation``, ``quote_is_outdated`` etc.)."""
    data = _cached_fetch(["broker", "holdings", "--json"], "holdings")
    if data is None:
        return None
    result = data.get("result") or {}
    return result.get("items")


def reset_cache() -> None:
    """Force the next ``overview()`` / ``holdings()`` call to re-fetch."""
    global _overview_cache, _holdings_cache
    with _payload_lock:
        _overview_cache = None
        _holdings_cache = None


# ---------------------------------------------------------------------------
# Lookup helpers for the rest of the codebase
# ---------------------------------------------------------------------------

def live_mid_by_isin() -> dict[str, float] | None:
    """Quick ``{isin: mid_price_eur}`` map from ``holdings()``. Returns
    ``None`` if the API isn't available so callers can fall back."""
    items = holdings()
    if items is None:
        return None
    return {h["isin"]: float(h["quote_mid_price"]) for h in items
            if h.get("quote_mid_price") is not None
            and h.get("quote_currency", "EUR").upper() == "EUR"}


def fifo_price_by_isin() -> dict[str, float] | None:
    """Scalable's own FIFO cost basis per ISIN — useful as a cross-check
    against our own FIFO engine."""
    items = holdings()
    if items is None:
        return None
    return {h["isin"]: float(h["fifo_price"]) for h in items
            if h.get("fifo_price") is not None}


_PERIOD_TO_API_KEY = {
    "1D": "INTRADAY",
    "1W": "ONE_WEEK",
    "1M": "ONE_MONTH",
    "3M": "THREE_MONTHS",
    "6M": "SIX_MONTHS",
    "YTD": "YEAR_TO_DATE",
    "1Y": "ONE_YEAR",
    "MAX": "MAX",
}


def absolute_return_by_period() -> dict[str, float] | None:
    """``{dashboard_period_label: euros}`` from Scalable's own performance
    array. Maps the API's ``ONE_WEEK`` / ``YEAR_TO_DATE`` etc. names back
    to the dashboard's button labels (``1W``, ``YTD``, ...)."""
    o = overview()
    if o is None:
        return None
    out: dict[str, float] = {}
    by_key = {p["timeframe"]: p for p in (o.get("performance") or [])}
    for dash_key, api_key in _PERIOD_TO_API_KEY.items():
        p = by_key.get(api_key)
        if p is not None and p.get("simpleAbsoluteReturn") is not None:
            out[dash_key] = float(p["simpleAbsoluteReturn"])
    return out or None


def valuation_total() -> float | None:
    """Total portfolio EUR value (securities + crypto + cash) from Scalable."""
    o = overview()
    if o is None:
        return None
    val = o.get("valuation") or {}
    if val.get("total") is None:
        return None
    return float(val["total"])


_PENDING_STATUSES = {"CREATED", "REQUESTED", "PENDING", "PARTIAL_FILLED",
                     "CANCEL_REQUESTED"}


_ALLOC_NICE_NAME = {
    "PRODUCT_TYPE": "Product type",
    "ASSET_CLASS": "Asset class",
    "EQUITY_SECTOR": "Equity sector",
    "FIXED_INCOME_SECTOR": "Fixed-income sector",
    "REGION": "Region",
    "COUNTRY": "Country",
}


def _label_for_bucket(pos: dict) -> str:
    return (pos.get("name") or pos.get("label")
            or pos.get("id", "").split("-")[-1] or "other").replace("_", " ")


def allocation_breakdowns() -> dict[str, list[dict]] | None:
    """Return Scalable's pre-computed allocation pies as a dict keyed by
    breakdown name (PRODUCT_TYPE, ASSET_CLASS, EQUITY_SECTOR, REGION,
    ...). Each value is a list of ``{label, value_eur, weight}`` rows
    ready to plot. ``None`` if the API isn't available."""
    if not is_available():
        return None
    data = _run_sc_json(["broker", "analytics", "--json"])
    if not data:
        return None
    out: dict[str, list[dict]] = {}
    for buckets in (data.get("result") or {}).get("allocations") or []:
        bid = buckets.get("id", "")
        # ID looks like "<portfolio>-Allocations-<TYPE>"
        key = bid.rsplit("-", 1)[-1] if "-" in bid else bid
        rows = []
        for pos in buckets.get("positions") or []:
            weight = float(pos.get("weight") or 0.0)
            val_obj = pos.get("valuation") or {}
            val_eur = (float(val_obj.get("amount"))
                       if isinstance(val_obj, dict)
                       and val_obj.get("amount") is not None
                       else None)
            rows.append({
                "label": _label_for_bucket(pos).title(),
                "value_eur": val_eur,
                "weight": weight,
            })
        if rows:
            out[key] = rows
    return out or None


def allocation_nice_name(key: str) -> str:
    return _ALLOC_NICE_NAME.get(key, key.replace("_", " ").title())


def pending_orders(limit: int = 25) -> list[dict] | None:
    """Open / unsettled orders that haven't shown up in the CSV yet."""
    if not is_available():
        return None
    data = _run_sc_json(["broker", "transactions", "--page-size", str(limit),
                         "--json"])
    if not data:
        return None
    items = (data.get("result") or {}).get("items") or []
    out: list[dict] = []
    for it in items:
        st = (it.get("status") or "").upper()
        if st not in _PENDING_STATUSES:
            continue
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# Login: spawn ``sc login`` in the background and capture its URL
# ---------------------------------------------------------------------------

# Common URL patterns the device-flow CLI is likely to emit.
_URL_RE = re.compile(
    r"https://[a-zA-Z0-9./?=#&_\-:%]+(?:scalable|sc\-app|login|auth)"
    r"[a-zA-Z0-9./?=#&_\-:%]*",
    re.IGNORECASE,
)

# Module-level handle to the latest login process so we can poll / cancel.
_login_proc: subprocess.Popen | None = None
_login_url: str | None = None
_login_lock = threading.Lock()


def start_login() -> str:
    """Spawn ``sc login`` and scrape its stdout for the OAuth device-code
    URL. Returns the captured URL (preferred), or an empty string if we
    couldn't parse one — in which case the UI shows manual instructions.
    """
    global _login_proc, _login_url
    binary = _sc_path()
    if binary is None:
        return ""
    with _login_lock:
        # If a previous attempt is still running, reuse its URL.
        if (_login_proc is not None and _login_proc.poll() is None
                and _login_url):
            return _login_url
        try:
            _login_proc = subprocess.Popen(
                [binary, "login"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except (OSError, FileNotFoundError):
            return ""

    # Read stdout for up to a few seconds, looking for the auth URL.
    proc = _login_proc
    captured: list[str] = []
    found: str = ""
    deadline = time.time() + 6.0

    def _reader():
        nonlocal found
        try:
            for line in proc.stdout:           # type: ignore[union-attr]
                captured.append(line)
                m = _URL_RE.search(line)
                if m and not found:
                    found = m.group(0).rstrip(".,;:)")
                    break
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    while time.time() < deadline and not found:
        time.sleep(0.1)

    with _login_lock:
        _login_url = found
    return found


def login_finished_successfully() -> bool:
    """Has ``sc login`` exited with a valid session? Called by the modal's
    poll to know when to dismiss itself."""
    with _login_lock:
        proc = _login_proc
    if proc is None:
        return False
    if proc.poll() is None:
        return False
    # Process exited — re-check whoami to confirm session is actually live.
    return status(force=True).state == "connected"


def cancel_login() -> None:
    """Kill any in-flight login subprocess; used when the user closes the
    modal without completing."""
    global _login_proc, _login_url
    with _login_lock:
        if _login_proc is not None and _login_proc.poll() is None:
            try:
                _login_proc.terminate()
            except Exception:
                pass
        _login_proc = None
        _login_url = None
