"""Load Scalable Capital broker transaction CSV exports.

The CSV uses ``;`` as separator and German number formatting where ``.`` is the
thousands separator and ``,`` is the decimal separator. Numeric columns can be
empty for cash-only rows.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import pandas as pd
from zoneinfo import ZoneInfo

BERLIN_TZ = ZoneInfo("Europe/Berlin")

TX_DIR = os.path.join(os.path.dirname(__file__), "..", "transaction_data")

# Transaction types we explicitly recognise (kept here for clarity).
SECURITY_BUY_TYPES = {"Buy", "Savings plan", "Reinvestment_Distribution"}
SECURITY_SELL_TYPES = {"Sell"}
CASH_FLOW_IN_TYPES = {"Deposit", "Cash Transfer In"}
CASH_FLOW_OUT_TYPES = {"Withdrawal", "Cash Transfer Out"}
CASH_INTEREST_TYPES = {"Interest"}
CASH_DIVIDEND_TYPES = {"Distribution"}
IGNORED_TYPES = {"Taxes"}


@dataclass
class LoadedTransactions:
    raw: pd.DataFrame          # everything, executed only
    securities: pd.DataFrame   # security side (buys/sells/distributions in shares)
    cash: pd.DataFrame         # cash-side movements
    file_path: str
    file_timestamp: datetime


_NUM_RE = re.compile(r"^-?\d{1,3}(?:\.\d{3})*(?:,\d+)?$|^-?\d+(?:,\d+)?$")


def _parse_german_number(value) -> float:
    """Convert a German-formatted number string to float. Returns NaN if empty."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return float("nan")
    # Remove thousands separators, swap decimal comma to dot.
    if _NUM_RE.match(s):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def find_latest_csv(directory: str = TX_DIR) -> str:
    """Return the newest CSV in the transaction_data directory.

    Files are named ``YYYY-MM-DD_HH-MM-SS_Scalable_Capital_*.csv`` so a
    lexicographic sort picks the most recent export.
    """
    pattern = os.path.join(directory, "*_Scalable_Capital_*Transactions*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Scalable Capital transactions CSV found in {directory}"
        )
    return files[-1]


def _extract_export_timestamp(path: str) -> datetime:
    """Parse the export timestamp from a filename like
    ``2026-06-09_12-23-59_Scalable_Capital_Scalable_Broker_Transactions.csv``."""
    name = os.path.basename(path)
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", name)
    if not m:
        return datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.strptime(f"{m.group(1)} {m.group(2).replace('-', ':')}",
                             "%Y-%m-%d %H:%M:%S")


def load_transactions(path: str | None = None) -> LoadedTransactions:
    """Load and normalise a Scalable Capital transactions CSV."""
    path = path or find_latest_csv()
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Strip stray whitespace and quotes.
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    df = df[df["status"] == "Executed"].copy()

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for col in ("shares", "price", "amount", "fee", "tax"):
        df[col] = df[col].apply(_parse_german_number)

    df["isin"] = df["isin"].replace("", pd.NA)
    df["description"] = df["description"].replace("", pd.NA)

    securities = df[df["assetType"] == "Security"].copy()
    cash = df[df["assetType"] == "Cash"].copy()

    return LoadedTransactions(
        raw=df,
        securities=securities,
        cash=cash,
        file_path=path,
        file_timestamp=_extract_export_timestamp(path),
    )


def isin_descriptions(tx: pd.DataFrame) -> dict[str, str]:
    """Map each ISIN to its most-recently-seen description."""
    sub = tx.dropna(subset=["isin", "description"]).sort_values("datetime")
    return dict(zip(sub["isin"], sub["description"]))


# ---------------------------------------------------------------------------
# Optional augmentation with transactions from the Scalable API
# ---------------------------------------------------------------------------

def _utc_str_to_berlin_naive(utc_str: str) -> pd.Timestamp:
    """Convert an API last_event_datetime (ISO-8601, UTC) into a Berlin-local
    naive ``Timestamp`` so it lines up with the CSV's German local times."""
    ts = pd.Timestamp(utc_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(BERLIN_TZ).tz_localize(None)


def _api_item_to_csv_row(item: dict, csv_type: str) -> dict | None:
    """Translate one API transaction record into the CSV-shape row used by
    ``load_transactions``. Returns ``None`` for unmapped items."""
    is_security = item.get("type") == "SECURITY_TRANSACTION"
    when_utc = item.get("last_event_datetime")
    if not when_utc:
        return None
    when_berlin = _utc_str_to_berlin_naive(when_utc)
    amount = item.get("amount")
    quantity = item.get("quantity")
    isin = item.get("isin") if is_security else item.get("related_isin")
    description = item.get("description") or ""

    if is_security:
        shares = float(quantity) if quantity not in (None, "") else float("nan")
        # API summary doesn't expose fee/tax — leave as 0 so the FIFO
        # engine doesn't double-deduct anything we already have in CSV.
        price = (abs(float(amount)) / shares
                 if amount not in (None, "") and shares else float("nan"))
        # CSV sign convention: Buy/Savings plan/Reinvestment_Distribution
        # have negative cash impact; Sell positive. API already follows
        # that, so we don't flip signs.
    else:
        shares = float("nan")
        price = float("nan")

    return {
        "date": when_berlin.strftime("%Y-%m-%d"),
        "time": when_berlin.strftime("%H:%M:%S"),
        "status": "Executed",
        "reference": item.get("id") or "",
        "description": description or pd.NA,
        "assetType": "Security" if is_security else "Cash",
        "type": csv_type,
        "isin": isin or pd.NA,
        "shares": shares,
        "price": price,
        "amount": float(amount) if amount not in (None, "") else float("nan"),
        "fee": 0.0,
        "tax": 0.0,
        "currency": item.get("currency") or "EUR",
        "datetime": when_berlin,
    }


def augment_with_api_transactions(tx: LoadedTransactions,
                                   api_items: list[dict]) -> tuple[LoadedTransactions, int]:
    """Merge new transactions from the Scalable API into the CSV-derived
    ``LoadedTransactions``. Returns ``(new_tx, n_added)``.

    Only API items strictly newer than the CSV's latest ``datetime`` are
    appended — anything overlapping is assumed already present in the CSV.
    """
    # Late import so this module stays usable without the API dependency.
    from . import scalable_api

    if not api_items:
        return tx, 0

    csv_max = tx.raw["datetime"].max()

    # CSV timestamps have second precision; the API returns milliseconds.
    # A transaction in the API at e.g. 19:46:58.594 is the same event the
    # CSV exports at 19:46:58 — including it again would double-count.
    # Round the cutoff up to the next second and require strict >.
    if pd.notna(csv_max):
        cutoff = (pd.Timestamp(csv_max).floor("s")
                  + pd.Timedelta(seconds=1))
    else:
        cutoff = None

    # Also build a set of (isin, side, quantity, second) keys already in
    # the CSV so we catch overlap on identical events the cutoff misses.
    csv_keys: set[tuple] = set()
    for r in tx.raw.itertuples(index=False):
        if r.assetType != "Security":
            continue
        key = (r.isin or "", str(r.type or ""),
               round(float(r.shares), 6) if pd.notna(r.shares) else None,
               pd.Timestamp(r.datetime).floor("s"))
        csv_keys.add(key)

    new_rows: list[dict] = []
    for item in api_items:
        when_utc = item.get("last_event_datetime")
        if not when_utc:
            continue
        when_berlin = _utc_str_to_berlin_naive(when_utc)
        if cutoff is not None and when_berlin < cutoff:
            continue  # already covered by the CSV export
        csv_type = scalable_api._csv_type(item)
        if csv_type is None:
            continue
        row = _api_item_to_csv_row(item, csv_type)
        if row is None:
            continue
        # Second-pass dedup by event identity.
        if row["assetType"] == "Security":
            key = (
                row["isin"] or "",
                row["type"],
                round(float(row["shares"]), 6) if pd.notna(row["shares"])
                else None,
                pd.Timestamp(row["datetime"]).floor("s"),
            )
            if key in csv_keys:
                continue
        new_rows.append(row)

    if not new_rows:
        return tx, 0

    extra = pd.DataFrame(new_rows)
    # Align columns to match the CSV-loaded shape.
    for col in tx.raw.columns:
        if col not in extra.columns:
            extra[col] = pd.NA
    extra = extra[tx.raw.columns]

    combined = (pd.concat([tx.raw, extra], ignore_index=True)
                .sort_values("datetime")
                .reset_index(drop=True))
    securities = combined[combined["assetType"] == "Security"].copy()
    cash = combined[combined["assetType"] == "Cash"].copy()
    new_tx = replace(tx, raw=combined, securities=securities, cash=cash)
    return new_tx, len(new_rows)
