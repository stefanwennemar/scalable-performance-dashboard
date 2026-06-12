# CLAUDE.md — guidance for Claude Code

This project is a personal portfolio dashboard for a Scalable Capital broker
account. The user is the account holder. The dashboard reads a CSV export of
the user's transaction history (in `transaction_data/`), fetches live quotes
from gettex.de, and renders a Dash web UI.

## Environment

- Python 3.12 (uv-managed). Use `uv run python …` and `uv add <pkg>`; do not
  invoke pip directly or modify `.venv` by hand.
- Launch the server with `uv run python run.py`.
- The CSV input is German-formatted: `;` separator, `1.234,56` numbers,
  `YYYY-MM-DD` dates. See `dashboard/data_loader.py` for the canonical parser.
- Live prices come from gettex.de via Playwright (headless Chromium). The
  page is fully client-side (SAML + WebSocket), so there is no public REST
  endpoint — the scraper waits for `document.title` to populate.
- Historical prices come from yfinance, with a fallback to interpolating
  transaction prices for ISINs Yahoo doesn't index.

## Source layout

```
dashboard/
    data_loader.py       CSV parse, German number format, exec-only filter
    portfolio.py         FIFO engine — positions, lots, realized P&L, cash
    prices.py            gettex.de live bid/ask via Playwright
    historical_prices.py yfinance daily closes, ISIN→ticker, FX
    performance.py       Daily value panel, TWR, simple return, XIRR, risk
    state.py             Process-wide state cache (thread-safe lazy loads)
    app.py               Dash UI + callbacks
assets/styles.css        Scalable-style dark theme (emerald accent #28EBCF)
transaction_data/        Drop CSV exports here; newest by name wins
cache/                   On-disk caches; safe to delete
run.py                   Entry point
```

## Transaction-type cheat sheet

Drawn from the Scalable CSV — see `memory/scalable_csv_format.md` for the
full list. Highlights:

- `Buy` / `Savings plan` / `Reinvestment_Distribution` add shares, decrement
  cash. Cost basis includes fees (= `abs(amount)/shares`).
- `Sell` consumes lots FIFO; realized P&L = proceeds − matched cost basis.
- `Distribution` (Cash) = dividend/coupon — part of the security's total
  return, **not** an external cash flow.
- `Deposit`, `Withdrawal`, `Cash Transfer In/Out` are the only true external
  cash flows used in TWR / XIRR denominators.
- `Security transfer` (in or out) is in-kind, but for TWR purposes must
  count as an external flow valued at `shares * price` — otherwise an
  in-kind transfer looks like free portfolio performance.
- `Corporate action` with negative shares at price ~€0.001 is a worthless
  write-off (treat as a sell at that price). Paired share-for-share
  exchanges between two ISINs net to zero.
- `Taxes` (Cash) — small ad-hoc tax adjustments — are ignored.

## Conventions worth keeping

- The historical value panel uses **a single price source per day** (Yahoo)
  so today-vs-yesterday isn't polluted by Yahoo-vs-gettex drift. Live gettex
  mid is layered on top only in user-facing snapshot cells (portfolio
  value KPI, positions table).
- Today's change is computed as
  `value[today] - value[yesterday] - external_flow[today]` so a withdrawal
  today doesn't look like a portfolio loss.
- Simple return = `(V_end − V_start − net_flow) / (V_start + max(net_flow, 0))`
  — net flow, not gross inflows, so in/out churn doesn't double-count the
  denominator.
- The performance chart's three modes are **TWR / Simple / EUR**. EUR is the
  cumulative euro gain net of external flows, anchored to 0 at window start.
- Dash pattern-matching IDs use `{"type": <name>, "isin": <isin>}` etc. for
  drilldown rows.

## Don'ts

- Don't add a Yahoo override to "today" in `build_value_panel`. The chart and
  TWR must stay on a single price source.
- Don't count `Distribution`, `Reinvestment_Distribution`, or `Interest` as
  external cash flows — they are part of returns.
- Don't drop the FIFO model in favour of average cost; the realized-P&L tab
  and the open-lot drilldown depend on per-lot tracking.
- Don't introduce blocking work on the Dash main thread; Playwright must run
  in its own thread (see `prices.fetch_live_prices`).
