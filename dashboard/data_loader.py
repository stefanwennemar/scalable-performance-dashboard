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


def _num(v) -> float | None:
    return float(v) if v not in (None, "") else None


def _trade_from_details(detail: dict | None, summary_amount: float
                        ) -> tuple[float, float, float, float] | None:
    """``(amount, price, fee, tax)`` in CSV convention from a
    ``security_trade`` details payload, or ``None`` if unusable.

    CSV convention: ``amount`` is the gross market value (negative for
    buys), fee and tax are separate positive numbers. The result is only
    accepted if it reproduces the summary's net cash amount.
    """
    trade = (detail or {}).get("security_trade") or {}
    tta = trade.get("trade_transaction_amounts") or {}
    gross = _num(tta.get("market_valuation"))
    price = _num(trade.get("average_price"))
    if gross is None or price is None:
        return None
    fee = sum(_num(tta.get(k)) or 0.0 for k in
              ("transaction_fee", "venue_fee", "crypto_spread_fee"))
    tax = _num(tta.get("tax_amount")) or 0.0
    if summary_amount < 0:
        # Buy side: fold any transaction tax (e.g. French FTT) into the
        # fee so it lands in the lot's cost basis and in cash.
        amount, fee, tax = -gross, fee + tax, 0.0
    else:
        amount = gross
    if abs((amount - fee - tax) - summary_amount) > 0.015:
        return None
    return amount, price, fee, tax


def _api_item_to_csv_row(item: dict, csv_type: str,
                         detail: dict | None = None) -> dict | None:
    """Translate one API transaction record into the CSV-shape row used by
    ``load_transactions``. Returns ``None`` for unmapped items.

    ``detail`` is the optional ``sc broker transaction details`` payload;
    it supplies the fee/tax split the summary record lacks."""
    api_type = item.get("type")
    is_security = api_type in ("SECURITY_TRANSACTION",
                               "NON_TRADE_SECURITY_TRANSACTION")
    when_utc = item.get("last_event_datetime")
    if not when_utc:
        return None
    when_berlin = _utc_str_to_berlin_naive(when_utc)
    amount = item.get("amount")
    quantity = item.get("quantity")
    isin = item.get("isin") if is_security else item.get("related_isin")
    description = item.get("description") or ""
    fee = 0.0
    tax = 0.0

    if is_security:
        shares = float(quantity) if quantity not in (None, "") else float("nan")
        # Fallback when no details are available: the summary amount is
        # net cash (fees and tax already deducted), so treat it as the
        # gross with zero fee/tax. Cash stays right; P&L is off by tax.
        price = (abs(float(amount)) / shares
                 if amount not in (None, "") and shares else float("nan"))
        if api_type == "NON_TRADE_SECURITY_TRANSACTION":
            nt_type = (item.get("non_trade_security_transaction_type")
                       or "").upper()
            if nt_type.endswith("_OUT"):
                shares = -abs(shares)
        elif amount not in (None, ""):
            parsed = _trade_from_details(detail, float(amount))
            if parsed is not None:
                amount, price, fee, tax = parsed
    else:
        shares = float("nan")
        price = float("nan")
        # CSV Distribution rows carry the net cash in ``amount`` and the
        # withheld tax (negative = refund) in ``tax``; mirror that.
        tax_details = ((detail or {}).get("cash") or {}).get("tax_details")
        if tax_details and tax_details.get("tax_amount") is not None:
            tax = float(tax_details["tax_amount"])

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
        "fee": fee,
        "tax": tax,
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

    candidates: list[tuple[dict, str]] = []
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
        candidates.append((item, csv_type))

    details = scalable_api.transaction_details([
        item["id"] for item, csv_type in candidates
        if item.get("id") and (item.get("type") == "SECURITY_TRANSACTION"
                               or csv_type == "Distribution")
    ])

    new_rows: list[dict] = []
    for item, csv_type in candidates:
        row = _api_item_to_csv_row(item, csv_type, details.get(item.get("id")))
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


# ---------------------------------------------------------------------------
# Derivative knock-outs
# ---------------------------------------------------------------------------

# Leveraged products (turbos, warrants, factor certificates) never pay
# dividends. When one is knocked out, Scalable books two things:
#   1. a write-off of the remaining shares at €0.001 each (CSV: negative
#      "Corporate action"; API: SWAP_OUT) — sometimes missing entirely, and
#   2. a cash "Distribution" on the derivative's ISIN holding the €0.001
#      proceeds plus the tax refund triggered by the realised loss.
# Left as-is, the loss never gets realised (ghost open position) and the
# refund shows up as dividend income.
_DERIVATIVE_RE = re.compile(
    r"turbo|optionsschein|knock|faktor|\d+x\s+factor|mini[ -]?future|warrant"
    r"|\b(?:call|put)\b", re.I)
KNOCKOUT_PRICE = 0.001
_EPS = 1e-9


def is_derivative(description) -> bool:
    return isinstance(description, str) and bool(
        _DERIVATIVE_RE.search(description))


def normalize_derivative_knockouts(tx: LoadedTransactions
                                   ) -> tuple[LoadedTransactions, int]:
    """Rewrite knock-out events on derivative ISINs. Returns
    ``(new_tx, n_knockouts)``.

    - Negative-share "Corporate action" rows become ``Knock-out`` sells.
    - Each "Distribution" becomes a ``Knock-out`` cash row (the €0.001/share
      proceeds) plus a ``Tax refund`` cash row (the rest). If no write-off
      row preceded it, one is synthesised for the shares still open.
    - A write-off arriving after its refund is clipped to the shares still
      open, so it can't double-count.
    """
    raw = tx.raw.copy()
    deriv_isins = set(raw.loc[raw["description"].map(is_derivative),
                              "isin"].dropna())
    if not deriv_isins:
        return tx, 0

    # Security rows sort before cash rows at equal timestamps, so a
    # same-day write-off is seen before its refund.
    ordered = raw.assign(_cash=(raw["assetType"] == "Cash").astype(int)) \
        .sort_values(["datetime", "_cash"], kind="stable")

    drop: list = []
    extra: list[dict] = []
    running: dict[str, float] = {}
    n_knockouts = 0
    for idx, r in ordered.iterrows():
        isin = r["isin"]
        if pd.isna(isin) or isin not in deriv_isins:
            continue
        shares = r["shares"] if pd.notna(r["shares"]) else 0.0
        held = running.get(isin, 0.0)

        if r["assetType"] == "Security":
            if r["type"] in SECURITY_BUY_TYPES:
                running[isin] = held + shares
            elif r["type"] in SECURITY_SELL_TYPES:
                running[isin] = held - shares
            elif r["type"] == "Corporate action" and shares < -_EPS:
                qty = min(-shares, max(held, 0.0))
                if qty <= _EPS:
                    drop.append(idx)       # already written off
                    continue
                raw.loc[idx, "type"] = "Knock-out"
                raw.loc[idx, "shares"] = -qty
                running[isin] = held - qty
            elif r["type"] in ("Corporate action", "Security transfer"):
                running[isin] = held + shares
            continue

        if r["type"] != "Distribution":
            continue
        amount = r["amount"] if pd.notna(r["amount"]) else 0.0
        tax = r["tax"] if pd.notna(r["tax"]) else 0.0
        open_qty = max(held, 0.0)
        if tax:
            proceeds = amount + tax       # tax is negative for a refund
        else:
            proceeds = min(amount, round(open_qty * KNOCKOUT_PRICE, 2))
        refund = amount - proceeds

        if open_qty > _EPS:
            extra.append({**r.drop("_cash").to_dict(),
                          "assetType": "Security", "type": "Knock-out",
                          "shares": -open_qty,
                          "price": (proceeds / open_qty if proceeds > 0
                                    else KNOCKOUT_PRICE),
                          "amount": -proceeds, "fee": 0.0, "tax": 0.0})
            running[isin] = 0.0
        n_knockouts += 1

        base = {**r.drop("_cash").to_dict(), "fee": 0.0, "tax": 0.0}
        drop.append(idx)
        if abs(proceeds) > _EPS:
            extra.append({**base, "type": "Knock-out", "amount": proceeds})
        if abs(refund) > _EPS:
            extra.append({**base, "type": "Tax refund", "amount": refund})

    raw = raw.drop(index=drop)
    if extra:
        raw = pd.concat([raw, pd.DataFrame(extra)[raw.columns]],
                        ignore_index=True)
    # Some old write-offs quote €0.001/share but no cash ever arrived; book
    # those at zero so realized P&L matches the cash actually received.
    paid_isins = set(raw.loc[(raw["assetType"] == "Cash")
                             & raw["type"].isin(["Knock-out",
                                                 "Corporate action"]),
                             "isin"].dropna())
    unpaid = ((raw["assetType"] == "Security") & (raw["type"] == "Knock-out")
              & ~raw["isin"].isin(paid_isins))
    raw.loc[unpaid, ["price", "amount"]] = 0.0
    raw = raw.sort_values("datetime", kind="stable").reset_index(drop=True)
    return replace(tx, raw=raw,
                   securities=raw[raw["assetType"] == "Security"].copy(),
                   cash=raw[raw["assetType"] == "Cash"].copy()), n_knockouts


# ---------------------------------------------------------------------------
# Security-transfer round trips
# ---------------------------------------------------------------------------

def collapse_transfer_round_trips(tx: LoadedTransactions, max_days: int = 7
                                  ) -> tuple[LoadedTransactions, int]:
    """Drop "Security transfer" out/in pairs of the same ISIN and share
    count within ``max_days`` (e.g. a depot migration on 2025-12-05/06).

    Replayed literally, the out-leg discards the FIFO lots and the in-leg
    re-opens them at the transfer-day price, so every gain up to the
    transfer vanishes from realized *and* unrealized P&L. Returns
    ``(new_tx, n_pairs)``.
    """
    raw = tx.raw
    xfers = raw[raw["type"] == "Security transfer"].sort_values("datetime")
    outs = xfers[xfers["shares"] < 0]
    ins = xfers[xfers["shares"] > 0]
    used: set = set()
    drop: list = []
    for o_idx, o in outs.iterrows():
        cand = ins[(ins["isin"] == o["isin"])
                   & ((ins["shares"] + o["shares"]).abs() < 1e-6)
                   & (ins["datetime"] >= o["datetime"])
                   & (ins["datetime"] - o["datetime"]
                      <= pd.Timedelta(days=max_days))
                   & ~ins.index.isin(list(used))]
        if cand.empty:
            continue
        i_idx = cand.index[0]
        used.add(i_idx)
        drop += [o_idx, i_idx]
    if not drop:
        return tx, 0
    raw = raw.drop(index=drop).reset_index(drop=True)
    return replace(tx, raw=raw,
                   securities=raw[raw["assetType"] == "Security"].copy(),
                   cash=raw[raw["assetType"] == "Cash"].copy()), len(drop) // 2
