"""FIFO portfolio engine.

Replays the transaction history to produce:
- per-ISIN FIFO lot chains (open lots remaining today)
- realized P&L per sell event (FIFO matched)
- cash balance over time
- distribution income per ISIN

Conventions:
- Buys add lots at ``unit_cost = abs(amount) / shares`` (includes fees because
  Scalable Capital folds the fee into the amount).
- Sells consume from the oldest lot first.
- Reinvestment_Distribution behaves like a small buy.
- Savings plan behaves like a buy.
- Corporate action (Security) with negative shares behaves like a sell with the
  given (often near-zero) unit price — a worthless write-off.
- Corporate action (Security) with positive shares is treated as a position
  added at the given price (e.g. ticker switch).
- Security transfer (positive shares) is treated as a transfer-in at the stated
  price (used as the lot cost basis).
- Security transfer (negative shares) is treated as a transfer-out, removing
  shares FIFO with no realized P&L (the realised P&L of an outbound transfer is
  not knowable from this account alone).
- Cash flows (Deposit, Withdrawal, Cash Transfer In/Out, Distribution,
  Interest) update the cash balance only.
- Taxes are ignored entirely per the user's spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

EPS = 1e-9


@dataclass
class Lot:
    """A still-open FIFO purchase lot."""
    isin: str
    open_datetime: datetime
    shares: float          # remaining shares (after any partial sells)
    original_shares: float
    unit_cost: float       # EUR per share, fees included
    source_type: str       # transaction type that created it


@dataclass
class RealizedTrade:
    """A FIFO-matched sell (or write-off) producing realized P&L."""
    sell_datetime: datetime
    isin: str
    description: str | None
    shares: float
    sell_price: float
    proceeds: float        # gross proceeds (after fees absorbed in amount)
    cost_basis: float
    realized_pnl: float
    holding_days: int
    sell_type: str
    open_datetime: datetime  # earliest matched lot's open date


@dataclass
class Position:
    isin: str
    description: str | None
    shares: float
    avg_cost: float           # weighted avg cost of remaining lots
    cost_basis: float         # remaining_shares * avg_cost
    lots: list[Lot] = field(default_factory=list)


@dataclass
class PortfolioState:
    positions: dict[str, Position]
    realized: list[RealizedTrade]
    cash_balance: float
    cash_series: pd.DataFrame          # datetime, cash, delta, description
    distributions: pd.DataFrame        # per-ISIN cash distribution rows
    interest_total: float
    deposits_total: float              # sum of deposits + cash transfers in
    withdrawals_total: float           # sum of withdrawals + cash transfers out (positive)


def _consume_fifo(lots: list[Lot], shares_to_sell: float
                  ) -> tuple[float, datetime | None]:
    """Remove ``shares_to_sell`` from the oldest lots. Returns (total cost
    basis consumed, earliest open datetime of any consumed lot)."""
    remaining = shares_to_sell
    cost = 0.0
    earliest: datetime | None = None
    while remaining > EPS and lots:
        lot = lots[0]
        take = min(lot.shares, remaining)
        cost += take * lot.unit_cost
        if earliest is None:
            earliest = lot.open_datetime
        lot.shares -= take
        remaining -= take
        if lot.shares <= EPS:
            lots.pop(0)
    return cost, earliest


def build_portfolio(transactions: pd.DataFrame) -> PortfolioState:
    """Replay the (already executed-only) transactions in chronological order."""
    tx = transactions.sort_values("datetime").reset_index(drop=True)

    lots_by_isin: dict[str, list[Lot]] = {}
    desc_by_isin: dict[str, str] = {}
    realized: list[RealizedTrade] = []

    cash = 0.0
    cash_rows: list[dict] = []
    distributions: list[dict] = []
    interest_total = 0.0
    deposits_total = 0.0
    withdrawals_total = 0.0

    def add_lot(isin: str, dt, shares: float, unit_cost: float, src: str):
        lots_by_isin.setdefault(isin, []).append(Lot(
            isin=isin, open_datetime=dt, shares=shares,
            original_shares=shares, unit_cost=unit_cost, source_type=src,
        ))

    def record_realized(dt, isin, desc, shares, sell_price,
                        proceeds, cost_basis, open_dt, sell_type):
        holding_days = (dt - open_dt).days if open_dt is not None else 0
        realized.append(RealizedTrade(
            sell_datetime=dt, isin=isin, description=desc,
            shares=shares, sell_price=sell_price, proceeds=proceeds,
            cost_basis=cost_basis,
            realized_pnl=proceeds - cost_basis,
            holding_days=holding_days,
            sell_type=sell_type,
            open_datetime=open_dt or dt,
        ))

    for row in tx.itertuples(index=False):
        ttype = row.type
        amount = row.amount if pd.notna(row.amount) else 0.0
        shares = row.shares if pd.notna(row.shares) else 0.0
        price = row.price if pd.notna(row.price) else 0.0
        isin = row.isin if pd.notna(row.isin) else None
        desc = row.description if pd.notna(row.description) else None
        if isin and desc:
            desc_by_isin[isin] = desc

        if ttype == "Taxes":
            # Standalone tax adjustments (refunds or charges) — they are
            # actual cash movements, so they must update the cash balance,
            # but they should not count as portfolio performance.
            if row.assetType == "Cash" and amount != 0:
                cash += amount
                cash_rows.append({
                    "datetime": row.datetime, "cash": cash,
                    "delta": amount, "type": ttype, "description": desc,
                })
            continue

        if row.assetType == "Cash":
            fee = row.fee if pd.notna(row.fee) else 0.0
            if ttype in ("Deposit", "Cash Transfer In"):
                cash += amount - fee
                deposits_total += amount
            elif ttype in ("Withdrawal", "Cash Transfer Out"):
                cash += amount - fee  # amount is already negative
                withdrawals_total += -amount
            elif ttype == "Distribution":
                # Distribution amount is gross of withholding tax; the actual
                # cash movement is amount - tax.
                tax = row.tax if pd.notna(row.tax) else 0.0
                net = amount - tax
                cash += net
                if isin:
                    distributions.append({
                        "datetime": row.datetime,
                        "isin": isin,
                        "description": desc,
                        "amount": amount,    # gross — used for "income" stats
                        "tax": tax,
                        "net": net,
                    })
            elif ttype == "Interest":
                # Empirically the Interest "amount" appears to already be
                # the net cash movement (some rows pair amount=-0.13 with a
                # huge tax reclassification; subtracting tax would explode
                # the balance). Keep as-is.
                cash += amount
                interest_total += amount
            elif ttype == "Corporate action":
                cash += amount
            else:
                cash += amount
            cash_rows.append({
                "datetime": row.datetime, "cash": cash,
                "delta": amount, "type": ttype, "description": desc,
            })
            continue

        # Security side.
        if isin is None:
            continue

        if ttype in ("Buy", "Savings plan", "Reinvestment_Distribution"):
            # amount = -shares*price (does NOT include the broker fee).
            # Cash decreases by amount + fee. Cost basis = (price*shares + fee)
            # per share so the fee is folded into the lot's break-even price.
            if shares <= EPS:
                continue
            fee = row.fee if pd.notna(row.fee) else 0.0
            unit_cost = (abs(amount) + fee) / shares if shares else price
            add_lot(isin, row.datetime, shares, unit_cost, ttype)
            net_cash = amount - fee  # amount is negative, fee is positive
            cash += net_cash
            cash_rows.append({
                "datetime": row.datetime, "cash": cash,
                "delta": net_cash, "type": ttype, "description": desc,
            })

        elif ttype == "Sell":
            if shares <= EPS:
                continue
            lots = lots_by_isin.get(isin, [])
            cost_basis, earliest = _consume_fifo(lots, shares)
            # ``amount`` is gross proceeds (= shares * price). The actual
            # cash movement is amount - capital-gains tax - broker fee.
            # We report realised P&L gross of capital-gains tax (matching
            # Scalable's "PnL since inception" display) but net of the
            # broker fee since the fee genuinely lowered our profit.
            tax = row.tax if pd.notna(row.tax) else 0.0
            fee = row.fee if pd.notna(row.fee) else 0.0
            net_cash = amount - tax - fee
            record_realized(row.datetime, isin, desc, shares, price,
                            amount - fee, cost_basis, earliest, "Sell")
            cash += net_cash
            cash_rows.append({
                "datetime": row.datetime, "cash": cash,
                "delta": net_cash, "type": ttype, "description": desc,
            })

        elif ttype == "Corporate action":
            # negative shares = write-off / removal; positive = added shares
            if shares < -EPS:
                qty = -shares
                lots = lots_by_isin.get(isin, [])
                cost_basis, earliest = _consume_fifo(lots, qty)
                sell_price = abs(price) if price else 0.0
                proceeds = abs(amount) if amount else 0.0
                record_realized(row.datetime, isin, desc, qty, sell_price,
                                proceeds, cost_basis, earliest,
                                "Corporate action")
                # Cash effect of corporate action is recorded by the paired
                # cash row (if any); the security row carries no cash.
            elif shares > EPS:
                unit_cost = abs(amount) / shares if amount else price
                add_lot(isin, row.datetime, shares, unit_cost,
                        "Corporate action")

        elif ttype == "Security transfer":
            if shares > EPS:
                unit_cost = price if price > 0 else (
                    abs(amount) / shares if amount else 0.0)
                add_lot(isin, row.datetime, shares, unit_cost,
                        "Security transfer")
            elif shares < -EPS:
                _consume_fifo(lots_by_isin.get(isin, []), -shares)

    positions: dict[str, Position] = {}
    for isin, lots in lots_by_isin.items():
        open_lots = [l for l in lots if l.shares > EPS]
        if not open_lots:
            continue
        shares_total = sum(l.shares for l in open_lots)
        cost_total = sum(l.shares * l.unit_cost for l in open_lots)
        positions[isin] = Position(
            isin=isin,
            description=desc_by_isin.get(isin),
            shares=shares_total,
            avg_cost=cost_total / shares_total if shares_total else 0.0,
            cost_basis=cost_total,
            lots=open_lots,
        )

    cash_series = pd.DataFrame(cash_rows)
    if not cash_series.empty:
        cash_series = cash_series.sort_values("datetime").reset_index(drop=True)

    distributions_df = pd.DataFrame(distributions)

    return PortfolioState(
        positions=positions,
        realized=realized,
        cash_balance=cash,
        cash_series=cash_series,
        distributions=distributions_df,
        interest_total=interest_total,
        deposits_total=deposits_total,
        withdrawals_total=withdrawals_total,
    )


def positions_to_dataframe(positions: dict[str, Position]) -> pd.DataFrame:
    rows = []
    for p in positions.values():
        rows.append({
            "isin": p.isin,
            "description": p.description,
            "shares": p.shares,
            "avg_cost": p.avg_cost,
            "cost_basis": p.cost_basis,
        })
    if not rows:
        return pd.DataFrame(columns=["isin", "description", "shares",
                                     "avg_cost", "cost_basis"])
    return pd.DataFrame(rows).sort_values("cost_basis", ascending=False)


def realized_to_dataframe(realized: list[RealizedTrade]) -> pd.DataFrame:
    if not realized:
        return pd.DataFrame(columns=["sell_datetime", "isin", "description",
                                     "shares", "sell_price", "proceeds",
                                     "cost_basis", "realized_pnl",
                                     "holding_days", "sell_type"])
    rows = [r.__dict__ for r in realized]
    return pd.DataFrame(rows).sort_values("sell_datetime", ascending=False)
