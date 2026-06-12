# Scalable Performance Dashboard

A local, interactive performance dashboard for a **Scalable Capital** broker
account. Drop in your transaction CSV, get a clean web UI with live gettex
quotes, FIFO realised/unrealised P&L, time-weighted returns, risk metrics
and an MSCI ACWI benchmark overlay — without sending your data anywhere.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-emerald.svg)](LICENSE)
[![Built with Dash](https://img.shields.io/badge/built%20with-Dash-1.7+-success)](https://dash.plotly.com)

> **Privacy**: nothing is uploaded. The dashboard reads your CSV locally
> and serves the UI on `127.0.0.1` only. Live prices come from
> [gettex.de](https://www.gettex.de) (anonymous page scrape); historical
> prices for held positions and the benchmark from the same source, with
> [yfinance](https://github.com/ranaroussi/yfinance) as a fallback.

---

## Quick start

```bash
# 1. One-time tooling
curl -LsSf https://astral.sh/uv/install.sh | sh           # macOS / Linux
# or:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2. Project setup (downloads Python, deps, and headless Chromium)
uv sync
uv run playwright install chromium

# 3. Drop your CSV export into transaction_data/, then:
uv run python run.py
```

Open <http://127.0.0.1:8050> in your browser.

The first start takes **5–7 minutes** while live prices and historical
data are fetched for every open position. Subsequent starts are fast
(disk cache: 15 min for live prices, 12 h for historical).

### Sharing with a non-technical friend

Zip the repo (minus `cache/` and `transaction_data/*.csv`) and send it
along with **`How to install.html`** — a self-contained, double-clickable
walkthrough (in German) that picks the right instructions for macOS or
Windows. Friend just installs `uv`, then double-clicks
`Setup.command` / `Setup.bat`, drops in their CSV, and double-clicks
`Run Dashboard.command` / `Run Dashboard.bat`.

---

## What it shows

### Overview tab
- KPI row: portfolio value · cash · P&L (period, gross €) · Return TWR %
  · unrealised P&L · realised P&L (gross / net of tax)
- Performance chart with eight period buttons (1D / 1W / 1M / 3M / YTD /
  1Y / 3Y / MAX), three view modes (TWR / Simple / EUR) and a benchmark
  toggle (MSCI ACWI – IUSQ)
- Portfolio value chart with capital-invested overlay
- 20-metric grid: TWR · Simple · Annualised TWR · XIRR · Benchmark
  return · Alpha · Volatility · Max drawdown · Gross P&L · Net of tax ·
  Tax paid · Net external flow · Sharpe · Sortino · Calmar · Profit
  factor · Beta · R² · Information ratio · Tracking error
- Calendar-year breakdown table
- Open-positions table with click-to-drilldown FIFO lot chain

### Transactions tab
- FIFO-matched realised trades (every sell + corporate-action write-off)
- Full executed-transaction log (sortable + filterable)

### Best & Worst tab
- Top / bottom 25 single trades by realised P&L
- Top / bottom 25 positions by aggregated lifetime realised P&L

### Allocation tab
- Position weights and country/region pies
- Cash balance over time
- Distribution income per ISIN and net leverage interest

---

## Architecture

```
dashboard/
    data_loader.py        CSV parse (German number format, exec-only filter)
    portfolio.py          FIFO engine — positions, lots, realised P&L
    performance.py        Value panel, TWR, simple return, XIRR, risk
    benchmark.py          IUSQ history + beta / alpha / R² / IR / TE
    prices.py             gettex.de live bid/ask via Playwright/Chromium
    gettex_history.py     gettex.de daily closes via the same SAML session
    historical_prices.py  yfinance fallback (ISIN→symbol, EUR FX)
    state.py              Process-wide cache layer (thread-safe)
    app.py                Dash UI + callbacks
    open_browser.py       Auto-opens the browser when the server is ready

assets/
    styles.css            Scalable-style emerald-on-near-black theme
    animations.js         IntersectionObserver fade-ups + KPI count-up

transaction_data/         Drop your CSV exports here; newest by name wins
cache/                    On-disk caches (15 min live, 12 h historical)

How to install.html       Friend-friendly setup guide (German)
Setup.command/.bat        One-time setup launchers
Run Dashboard.command/.bat  Daily launchers
run.py                    Plain Python entry point
```

---

## Conventions worth knowing

- **Single price source for the panel** — the day-by-day value series
  uses one source per day (gettex EOD, falling back to Yahoo only if
  gettex has no listing). Live gettex mid quotes are layered on top
  only in user-facing snapshot cells (portfolio-value KPI, positions
  table). This avoids Yahoo-vs-gettex drift showing up as a fake
  intraday swing.
- **"Today's change"** is `value[today] − value[yesterday] −
  external_flow[today]`, so a withdrawal today doesn't look like a
  portfolio loss.
- **Sell tax** is reported in the CSV's `tax` column. Cash is
  decremented by `amount − tax`; the tax outflow is treated as an
  external flow in performance accounting, so reported P&L matches
  Scalable's gross "PnL since inception".
- **Fees** are baked into per-lot cost basis on buys, and netted off
  proceeds on sells.
- **Distribution** and **Interest** amounts are interpreted per CSV:
  Distribution gross of withholding (tax subtracted to get cash);
  Interest is already net.
- **`Taxes` rows** are real cash adjustments — they update the cash
  balance but are flagged as external flows so they don't affect
  performance.
- **YTD anchor** is 30 December of the previous year, matching Scalable
  app's convention. "Today" uses `Europe/Berlin` 09:00 (Xetra open) as
  the rollover, so US-evening views don't show tomorrow's date
  prematurely.

---

## Privacy & data flow

| What | Where it goes |
|---|---|
| Your CSV | Stays on disk, never sent off-machine. |
| Live quotes | HTTP GET to `gettex.de` only; one anonymous page hit per ISIN every ~15 min. |
| Historical closes | Same gettex endpoint (SAML handshake inside headless Chromium); yfinance fallback for non-listed ISINs. |
| The dashboard UI | Served on `127.0.0.1:8050` — no public bind, no auth needed. |

No telemetry, no analytics, no remote backend. If you `pip-audit` the
deps, you'll only see well-known libraries (Dash, Plotly, pandas,
yfinance, Playwright).

---

## Roadmap / nice-to-haves

- [ ] Optional FastAPI/Docker wrapper for hosted use
- [ ] User-pickable benchmark (currently hard-coded to IUSQ / MSCI ACWI)
- [ ] Compare two CSV exports side by side (deltas since last load)
- [ ] Export the metric grid to PDF

---

## Acknowledgements

Built with [Dash](https://dash.plotly.com), [Plotly](https://plotly.com/python/),
[pandas](https://pandas.pydata.org), [Playwright](https://playwright.dev/python/),
[yfinance](https://github.com/ranaroussi/yfinance) and the public
[gettex.de](https://www.gettex.de) widget. Visual palette inspired by
Scalable Capital's brand.

## License

[MIT](LICENSE) — do whatever you want, no warranty.
