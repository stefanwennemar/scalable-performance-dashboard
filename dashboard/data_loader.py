"""Load Scalable Capital broker transaction CSV exports.

The CSV uses ``;`` as separator and German number formatting where ``.`` is the
thousands separator and ``,`` is the decimal separator. Numeric columns can be
empty for cash-only rows.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

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
