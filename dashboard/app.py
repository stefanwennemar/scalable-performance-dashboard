"""Dash application entry point."""

from __future__ import annotations

import math
from datetime import datetime

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from . import state as ds
from .benchmark import BENCHMARK_NAME, compute_market_metrics
from .performance import (annualised_return, compute_metrics, max_drawdown,
                          simple_return, twr_series, window_start, xirr)
from .portfolio import Position

PERIODS = ["1D", "1W", "1M", "3M", "YTD", "1Y", "3Y", "MAX"]
DEFAULT_PERIOD = "YTD"
COLOR_GREEN = "#28EBCF"           # Scalable Capital emerald
COLOR_GREEN_FILL = "rgba(40, 235, 207, 0.14)"
COLOR_RED = "#ff5560"
COLOR_RED_FILL = "rgba(255, 85, 96, 0.14)"
COLOR_GREY = "#9aa0aa"
COLOR_LINE = "#28EBCF"

# Single source of truth for plotly font / hover styling — used by every
# chart on the dashboard so typography stays consistent with the rest of
# the UI.
_CHART_FONT = dict(
    family=("Inter, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', "
            "Arial, sans-serif"),
    color="#9aa0aa", size=12,
)
_HOVER_LABEL = dict(
    bgcolor="#16181d", bordercolor="#25282f",
    font=dict(family=_CHART_FONT["family"], size=12, color="#f4f4f6"),
)

# ---------------------------------------------------------------------------
# Initial data load (runs on import so Dash starts in a known state).
# ---------------------------------------------------------------------------
print("[startup] loading transactions...")
STATE = ds.load_state()
print(f"[startup] {len(STATE.portfolio.positions)} positions, "
      f"{len(STATE.portfolio.realized)} realized trades, "
      f"cash €{STATE.portfolio.cash_balance:,.2f}")

app = dash.Dash(
    __name__,
    title="Scalable Capital Performance",
    assets_folder="../assets",
    suppress_callback_exceptions=True,
)
server = app.server


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_eur(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"€{v:,.{decimals}f}"


def fmt_pct(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v * 100:+.{decimals}f}%"


def fmt_num(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:,.{decimals}f}"


def color_class(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "muted"
    return "pos" if v >= 0 else "neg"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def period_switch() -> html.Div:
    return html.Div([
        html.Div([
            html.Button(p, id={"type": "period-btn", "period": p},
                        className="period-btn" + (" active" if p == DEFAULT_PERIOD else ""),
                        n_clicks=0)
            for p in PERIODS
        ], className="period-switch"),
        html.Div([
            html.Button("TWR", id={"type": "mode-btn", "mode": "twr"},
                        className="mode-btn active", n_clicks=0),
            html.Button("Simple", id={"type": "mode-btn", "mode": "simple"},
                        className="mode-btn", n_clicks=0),
            html.Button("EUR", id={"type": "mode-btn", "mode": "eur"},
                        className="mode-btn", n_clicks=0),
        ], className="return-mode-switch"),
        html.Button(f"vs {BENCHMARK_NAME}", id="benchmark-toggle",
                    className="mode-btn", n_clicks=0,
                    style={"marginLeft": "12px", "padding": "6px 14px",
                           "border": "1px solid var(--border-subtle)",
                           "background": "var(--bg-elevated)",
                           "color": "var(--text-secondary)",
                           "borderRadius": "10px", "cursor": "pointer",
                           "fontSize": "12px"}),
    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
              "gap": "6px", "marginBottom": "12px"})


def kpi_cards_placeholder() -> html.Div:
    return html.Div(id="kpi-cards", className="kpi-row")


def overview_tab() -> html.Div:
    return html.Div([
        kpi_cards_placeholder(),

        html.Div([
            html.H3("Portfolio performance", className="card-title"),
            period_switch(),
            dcc.Loading(
                dcc.Graph(id="perf-chart", config={"displayModeBar": False}),
                type="circle", color=COLOR_GREEN,
            ),
        ], className="card"),

        html.Div([
            html.H3("Portfolio value", className="card-title"),
            html.P("Total account value over time (gettex EOD closes for "
                   "holdings + cash). Net deposits are overlaid for "
                   "reference.",
                   className="card-subtitle"),
            dcc.Loading(
                dcc.Graph(id="value-chart", config={"displayModeBar": False}),
                type="circle", color=COLOR_GREEN,
            ),
        ], className="card"),

        html.Div([
            html.H3("Performance & risk metrics", className="card-title"),
            html.Div(id="metrics-grid"),
        ], className="card"),

        html.Div([
            html.H3("Calendar-year breakdown", className="card-title"),
            html.P("Each row is one calendar year. The current year runs "
                   "until today.", className="card-subtitle"),
            html.Div(id="yearly-table"),
        ], className="card"),

        html.Div([
            html.H3("Open positions", className="card-title"),
            html.Div(id="positions-table"),
            html.Div(id="position-detail"),
        ], className="card"),
    ])


def returns_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.H3("Return analytics", className="card-title"),
            html.P("Period-by-period returns at daily, weekly or monthly "
                   "granularity, compared to the MSCI ACWI benchmark on "
                   "demand. The cards below add distribution, monthly "
                   "heatmap and drawdown views.",
                   className="card-subtitle"),
            # Window picker (shares store with the Overview-tab period buttons).
            html.Div([
                html.Span("Period:",
                          style={"color": "var(--text-secondary)",
                                 "fontSize": "12px",
                                 "marginRight": "8px"}),
                html.Div([
                    html.Button(p, id={"type": "ret-period-btn",
                                       "period": p},
                                className="period-btn"
                                + (" active" if p == DEFAULT_PERIOD else ""),
                                n_clicks=0)
                    for p in PERIODS
                ], className="period-switch"),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "12px"}),
            html.Div([
                # Granularity
                html.Div([
                    html.Span("Granularity:",
                              style={"color": "var(--text-secondary)",
                                     "fontSize": "12px",
                                     "marginRight": "8px"}),
                    html.Div([
                        html.Button("Daily",
                                    id={"type": "ret-gran-btn", "gran": "D"},
                                    className="period-btn", n_clicks=0),
                        html.Button("Weekly",
                                    id={"type": "ret-gran-btn", "gran": "W"},
                                    className="period-btn", n_clicks=0),
                        html.Button("Monthly",
                                    id={"type": "ret-gran-btn", "gran": "M"},
                                    className="period-btn active", n_clicks=0),
                    ], className="period-switch"),
                ], style={"display": "flex", "alignItems": "center"}),
                # Mode
                html.Div([
                    html.Span("Show as:",
                              style={"color": "var(--text-secondary)",
                                     "fontSize": "12px",
                                     "marginLeft": "20px",
                                     "marginRight": "8px"}),
                    html.Div([
                        html.Button("TWR %",
                                    id={"type": "ret-mode-btn", "mode": "pct"},
                                    className="mode-btn active", n_clicks=0),
                        html.Button("EUR",
                                    id={"type": "ret-mode-btn", "mode": "eur"},
                                    className="mode-btn", n_clicks=0),
                    ], className="return-mode-switch"),
                ], style={"display": "flex", "alignItems": "center"}),
                # Benchmark toggle
                html.Button(f"vs {BENCHMARK_NAME}",
                            id="ret-benchmark-toggle",
                            className="mode-btn", n_clicks=0,
                            style={"marginLeft": "20px",
                                   "padding": "6px 14px",
                                   "border": "1px solid var(--border-subtle)",
                                   "background": "var(--bg-elevated)",
                                   "color": "var(--text-secondary)",
                                   "borderRadius": "10px",
                                   "cursor": "pointer", "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center",
                      "gap": "6px", "flexWrap": "wrap"}),
        ], className="card"),

        html.Div([
            html.H3("Period returns", className="card-title"),
            dcc.Loading(
                dcc.Graph(id="ret-bars-chart",
                          config={"displayModeBar": False}),
                type="circle", color=COLOR_GREEN,
            ),
        ], className="card"),

        html.Div([
            html.H3("Period statistics", className="card-title"),
            html.Div(id="ret-stats-grid"),
        ], className="card"),

        html.Div([
            html.H3("Distribution of period returns",
                    className="card-title"),
            html.P("Where do returns cluster, and how fat are the tails?",
                   className="card-subtitle"),
            dcc.Loading(
                dcc.Graph(id="ret-hist-chart",
                          config={"displayModeBar": False}),
                type="circle", color=COLOR_GREEN,
            ),
        ], className="card"),

        html.Div([
            html.H3("Monthly TWR heatmap", className="card-title"),
            html.P("Each cell is one calendar month's time-weighted return. "
                   "Rows are years, columns are months.",
                   className="card-subtitle"),
            dcc.Loading(
                dcc.Graph(id="ret-heatmap-chart",
                          config={"displayModeBar": False}),
                type="circle", color=COLOR_GREEN,
            ),
        ], className="card"),

        html.Div([
            html.H3("Drawdown over time", className="card-title"),
            html.P("Percent below the running high-water mark — "
                   "peak-to-trough loss.", className="card-subtitle"),
            dcc.Loading(
                dcc.Graph(id="ret-drawdown-chart",
                          config={"displayModeBar": False}),
                type="circle", color=COLOR_GREEN,
            ),
        ], className="card"),
    ])


def transactions_tab() -> html.Div:
    return html.Div([
        # Pending-orders card (rendered as muted note when API is off or
        # there's nothing pending — render_pending_orders decides).
        html.Div(id="pending-orders-card"),
        html.Div([
            html.H3("Realized P&L (FIFO)", className="card-title"),
            html.P("Each row is one sell — including corporate-action write-offs "
                   "— matched against the oldest open lots.",
                   className="card-subtitle"),
            html.Div(id="realized-table"),
        ], className="card"),
        html.Div([
            html.H3("All executed transactions", className="card-title"),
            html.Div(id="transactions-table"),
        ], className="card"),
    ])


def best_worst_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.H3("Best individual trades", className="card-title"),
            html.P("Single sells (and corporate-action write-offs) ranked by "
                   "realized P&L. Gross of capital-gains tax.",
                   className="card-subtitle"),
            html.Div(id="best-trades-table"),
        ], className="card"),
        html.Div([
            html.H3("Worst individual trades", className="card-title"),
            html.P("The flipside — biggest realized losses.",
                   className="card-subtitle"),
            html.Div(id="worst-trades-table"),
        ], className="card"),
        html.Div([
            html.H3("Best by position (aggregated by ISIN)",
                    className="card-title"),
            html.P("All sells per ISIN summed together. A position whose "
                   "many small sells add up to a big win lands here.",
                   className="card-subtitle"),
            html.Div(id="best-by-isin-table"),
        ], className="card"),
        html.Div([
            html.H3("Worst by position (aggregated by ISIN)",
                    className="card-title"),
            html.Div(id="worst-by-isin-table"),
        ], className="card"),
    ])


def allocation_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.H3("Allocation", className="card-title"),
            html.Div(id="allocation-charts"),
        ], className="card"),
        html.Div([
            html.H3("Cash & deposits over time", className="card-title"),
            dcc.Graph(id="cash-chart", config={"displayModeBar": False}),
        ], className="card"),
        html.Div([
            html.H3("Distributions & dividends", className="card-title"),
            html.Div(id="dividends-summary"),
        ], className="card"),
    ])


app.layout = html.Div([
    dcc.Store(id="active-period", data=DEFAULT_PERIOD),
    dcc.Store(id="return-mode", data="twr"),
    dcc.Store(id="benchmark-on", data=False),
    dcc.Store(id="selected-isin", data=None),
    dcc.Store(id="ret-granularity", data="M"),
    dcc.Store(id="ret-mode", data="pct"),
    dcc.Store(id="ret-benchmark-on", data=False),
    dcc.Store(id="api-modal-open", data=False),
    dcc.Store(id="api-login-url", data=None),
    # Polls the API status every 4 s while the connect modal is open so
    # the dashboard auto-closes the modal the moment login completes.
    dcc.Interval(id="api-status-poll", interval=4000, n_intervals=0,
                 disabled=True),
    dcc.Interval(id="refresh-trigger", interval=5 * 60 * 1000, n_intervals=0),

    html.Div([
        html.Div([
            html.H1("Scalable Performance Dashboard", className="app-title"),
            html.Div([
                f"Transactions exported "
                f"{STATE.tx.file_timestamp.strftime('%Y-%m-%d %H:%M')} · "
                f"{len(STATE.tx.raw):,} rows",
                html.Span(
                    f"  · +{STATE.api_added} pulled live from Scalable API"
                    if STATE.api_added else "",
                    style={"color": "var(--accent-green)"}),
            ], className="app-subtitle"),
        ]),
        html.Div([
            # Top row: API pill + refresh button on one horizontal line.
            html.Div([
                html.Div([
                    html.Span(className="dot"),
                    html.Span("Checking…", id="api-pill-text"),
                ], id="api-status-pill", className="api-pill", n_clicks=0),
                html.Button("Refresh prices", id="refresh-prices-btn",
                            className="period-btn", n_clicks=0,
                            style={"background": "var(--bg-elevated)",
                                   "color": "var(--text-primary)",
                                   "padding": "8px 16px",
                                   "borderRadius": "10px",
                                   "border": "none", "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center",
                      "justifyContent": "flex-end", "flexWrap": "wrap"}),
            # Sub-row: "Live prices: x/y fresh · updated …" sits below both.
            html.Div(id="prices-status",
                     style={"color": "var(--text-secondary)",
                            "fontSize": "11px", "marginTop": "6px",
                            "textAlign": "right"}),
        ], style={"display": "flex", "flexDirection": "column",
                  "alignItems": "stretch"}),
    ], className="app-header"),

    # Connect / install modal (lazy — empty children when closed).
    html.Div(id="api-modal", children=[]),
    # Hidden output target for the clientside window.open callback.
    html.Div(id="api-login-sink", style={"display": "none"}),

    dcc.Tabs(id="tabs", value="overview", parent_className="dash-tabs",
             className="dash-tabs", children=[
        dcc.Tab(label="Overview", value="overview", className="dash-tab",
                selected_className="dash-tab--selected", children=overview_tab()),
        dcc.Tab(label="Returns", value="returns", className="dash-tab",
                selected_className="dash-tab--selected", children=returns_tab()),
        dcc.Tab(label="Transactions", value="transactions",
                className="dash-tab", selected_className="dash-tab--selected",
                children=transactions_tab()),
        dcc.Tab(label="Best & Worst", value="best_worst",
                className="dash-tab", selected_className="dash-tab--selected",
                children=best_worst_tab()),
        dcc.Tab(label="Allocation", value="allocation", className="dash-tab",
                selected_className="dash-tab--selected", children=allocation_tab()),
    ]),
], className="app-shell")


# ---------------------------------------------------------------------------
# Callbacks: period / mode buttons
# ---------------------------------------------------------------------------

@app.callback(
    Output("active-period", "data"),
    Input({"type": "period-btn", "period": dash.ALL}, "n_clicks"),
    Input({"type": "ret-period-btn", "period": dash.ALL}, "n_clicks"),
    State("active-period", "data"),
)
def update_active_period(_ov_clicks, _ret_clicks, current):
    """Either tab's period buttons write to the same shared store, so the
    active window stays consistent across the Overview and Returns tabs."""
    ctx = callback_context
    if not ctx.triggered or not (any(_ov_clicks) or any(_ret_clicks)):
        return current or DEFAULT_PERIOD
    import json
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    return json.loads(trig)["period"]


@app.callback(
    Output({"type": "period-btn", "period": dash.ALL}, "className"),
    Output({"type": "ret-period-btn", "period": dash.ALL}, "className"),
    Input("active-period", "data"),
)
def sync_period_button_classes(period):
    """Highlight the active period button on *both* tabs."""
    classes = ["period-btn" + (" active" if p == period else "")
               for p in PERIODS]
    return classes, classes


@app.callback(
    Output("return-mode", "data"),
    Output({"type": "mode-btn", "mode": dash.ALL}, "className"),
    Input({"type": "mode-btn", "mode": dash.ALL}, "n_clicks"),
    State("return-mode", "data"),
)
def update_return_mode(_clicks, current):
    ctx = callback_context
    modes = ["twr", "simple", "eur"]
    if not ctx.triggered or not any(_clicks):
        mode = current or "twr"
    else:
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        import json
        mode = json.loads(trig)["mode"]
    classes = ["mode-btn" + (" active" if m == mode else "") for m in modes]
    return mode, classes


@app.callback(
    Output("benchmark-on", "data"),
    Output("benchmark-toggle", "className"),
    Output("benchmark-toggle", "style"),
    Input("benchmark-toggle", "n_clicks"),
    State("benchmark-on", "data"),
)
def update_benchmark_toggle(n_clicks, current):
    new_state = (not bool(current)) if n_clicks else False
    cls = "mode-btn active" if new_state else "mode-btn"
    style = {"marginLeft": "12px", "padding": "6px 14px",
             "borderRadius": "10px", "cursor": "pointer", "fontSize": "12px",
             "border": "1px solid var(--border-subtle)"}
    if new_state:
        style.update({"background": "var(--accent-green)",
                      "color": "#0a0a0a", "fontWeight": "600",
                      "borderColor": "transparent"})
    else:
        style.update({"background": "var(--bg-elevated)",
                      "color": "var(--text-secondary)"})
    return new_state, cls, style


# ---------------------------------------------------------------------------
# Scalable-API status pill + connect modal
# ---------------------------------------------------------------------------

from . import scalable_api as _scalable_api


_PILL_BY_STATE = {
    "connected": ("connected",
                  "Scalable API · connected",
                  "Live prices, today's change and valuations come straight "
                  "from your Scalable account."),
    "logged_out": ("connectable",
                   "Scalable API · connect",
                   "Click to connect for live prices straight from "
                   "Scalable Capital."),
    "missing": ("missing",
                "Scalable API · install for live data",
                "Click for setup instructions."),
    "error": ("error",
              "Scalable API · reconnect",
              "Click to reconnect."),
}


@app.callback(
    Output("api-status-pill", "className"),
    Output("api-pill-text", "children"),
    Output("api-status-pill", "title"),
    Input("prices-status", "children"),       # triggered on every refresh
    Input("api-status-poll", "n_intervals"),  # active while modal is open
)
def render_api_pill(_status, _ticks):
    s = _scalable_api.status()
    cls, label, tip = _PILL_BY_STATE.get(
        s.state, ("error", "Scalable API · error", s.detail or ""))
    return f"api-pill {cls}", label, tip


@app.callback(
    Output("api-modal-open", "data"),
    Output("api-status-poll", "disabled"),
    Output("api-login-url", "data"),
    Input("api-status-pill", "n_clicks"),
    Input({"type": "api-modal-btn", "action": dash.ALL}, "n_clicks"),
    Input("api-status-poll", "n_intervals"),
    State("api-modal-open", "data"),
)
def manage_api_modal(_pill_clicks, _btn_clicks, _ticks, is_open):
    """Manage modal open/close + ``sc login`` lifecycle.

    Trigger map:
    - clicking the pill opens the modal (if not already connected)
    - the modal's "Connect now" action spawns ``sc login`` and writes
      the captured OAuth URL into the api-login-url Store — a clientside
      callback then opens it via window.open(), and the modal also
      renders it as a clickable link in case the popup is blocked
    - the "Cancel" action kills the login process and closes the modal
    - the poll interval auto-closes the modal once whoami says we're in
    """
    ctx = callback_context
    trig = ctx.triggered_id

    # Auto-close when login completes.
    if (is_open and isinstance(trig, str)
            and trig == "api-status-poll"
            and _scalable_api.is_available()):
        return False, True, None

    # Pill click → open the modal (unless we're already connected, in
    # which case the pill is just informational).
    if trig == "api-status-pill":
        s = _scalable_api.status(force=True)
        if s.state == "connected":
            return False, True, None
        return True, False, None

    # Modal action button.
    if isinstance(trig, dict) and trig.get("type") == "api-modal-btn":
        action = trig.get("action")
        if action == "cancel":
            _scalable_api.cancel_login()
            return False, True, None
        if action == "connect":
            url = _scalable_api.start_login()
            # The clientside callback below pops the URL in a new tab the
            # moment we write it; the modal also shows it as a clickable
            # fallback in case the browser blocked window.open().
            return True, False, (url or "")

    return is_open or False, not bool(is_open), dash.no_update


# Open the captured OAuth URL in a new tab on the user's side. Runs as a
# clientside JS callback so it's allowed to call window.open without a
# server round-trip.
app.clientside_callback(
    """
    function(url) {
        if (url && typeof url === 'string' && url.startsWith('http')) {
            window.open(url, '_blank', 'noopener,noreferrer');
        }
        return '';
    }
    """,
    Output("api-login-sink", "children"),
    Input("api-login-url", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("api-modal", "children"),
    Input("api-modal-open", "data"),
    Input("api-login-url", "data"),
)
def render_api_modal(is_open, login_url):
    if not is_open:
        return []
    s = _scalable_api.status()

    if s.state == "missing":
        body = html.Div([
            html.H3("Install the Scalable CLI"),
            html.P("The dashboard works without it, but if you install "
                   "Scalable's official CLI you'll get instant live prices "
                   "(no more 3-minute warmups), Scalable's own today's-change "
                   "value, and full coverage of derivatives."),
            html.P("On a Mac, run this once in Terminal:"),
            html.Pre("brew tap scalablecapital/sc && brew install sc"),
            html.P("On Windows / Linux see the install instructions at:"),
            html.Pre("https://github.com/ScalableCapital/scalable-cli"),
            html.P("After installing, re-open the dashboard and click the "
                   "status pill again to log in."),
            html.Div([
                html.Button("Got it", id={"type": "api-modal-btn",
                                          "action": "cancel"},
                            className="modal-btn primary", n_clicks=0),
            ], className="modal-actions"),
        ], className="modal-card")

    else:
        # logged_out / error / first-time connect
        if login_url and login_url.startswith("http"):
            # We've already spawned `sc login` and captured its OAuth URL.
            # The clientside callback also tries to pop it in a new tab;
            # the link below is the user-clickable fallback.
            body = html.Div([
                html.H3("Approve in your browser"),
                html.P("A new tab should have opened with the Scalable "
                       "login page. If not, click the button below."),
                html.A("Open Scalable login page",
                       href=login_url, target="_blank",
                       rel="noopener noreferrer",
                       className="modal-btn primary",
                       style={"textDecoration": "none",
                              "display": "inline-block",
                              "marginTop": "8px"}),
                html.P("Sign in and confirm the prompt on your phone. "
                       "The dashboard will switch to live data the "
                       "moment you approve — no need to come back here "
                       "manually.",
                       className="muted",
                       style={"marginTop": "14px"}),
                html.Div([
                    html.Button("Cancel", id={"type": "api-modal-btn",
                                              "action": "cancel"},
                                className="modal-btn", n_clicks=0),
                ], className="modal-actions"),
            ], className="modal-card")
        else:
            body = html.Div([
                html.H3("Connect to Scalable Capital"),
                html.P("Click below to start the device-code login. A "
                       "page will open in a new tab — sign in with your "
                       "Scalable account and confirm the prompt on your "
                       "phone. The dashboard will switch to live data "
                       "automatically the moment you approve."),
                html.P("Nothing is sent anywhere; the session lives only "
                       "on this computer.", className="muted"),
                html.Div([
                    html.Button("Cancel", id={"type": "api-modal-btn",
                                              "action": "cancel"},
                                className="modal-btn", n_clicks=0),
                    html.Button("Connect now",
                                id={"type": "api-modal-btn",
                                    "action": "connect"},
                                className="modal-btn primary", n_clicks=0),
                ], className="modal-actions"),
            ], className="modal-card")

    return html.Div(body, className="modal-backdrop")


# ---- Returns tab toggles --------------------------------------------------
@app.callback(
    Output("ret-granularity", "data"),
    Output({"type": "ret-gran-btn", "gran": dash.ALL}, "className"),
    Input({"type": "ret-gran-btn", "gran": dash.ALL}, "n_clicks"),
    State("ret-granularity", "data"),
)
def update_ret_gran(_clicks, current):
    ctx = callback_context
    grans = ["D", "W", "M"]
    if not ctx.triggered or not any(_clicks):
        g = current or "M"
    else:
        import json
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        g = json.loads(trig)["gran"]
    return g, ["period-btn" + (" active" if x == g else "") for x in grans]


@app.callback(
    Output("ret-mode", "data"),
    Output({"type": "ret-mode-btn", "mode": dash.ALL}, "className"),
    Input({"type": "ret-mode-btn", "mode": dash.ALL}, "n_clicks"),
    State("ret-mode", "data"),
)
def update_ret_mode(_clicks, current):
    ctx = callback_context
    modes = ["pct", "eur"]
    if not ctx.triggered or not any(_clicks):
        m = current or "pct"
    else:
        import json
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        m = json.loads(trig)["mode"]
    return m, ["mode-btn" + (" active" if x == m else "") for x in modes]


@app.callback(
    Output("ret-benchmark-on", "data"),
    Output("ret-benchmark-toggle", "className"),
    Output("ret-benchmark-toggle", "style"),
    Input("ret-benchmark-toggle", "n_clicks"),
    State("ret-benchmark-on", "data"),
)
def update_ret_benchmark_toggle(n_clicks, current):
    new_state = (not bool(current)) if n_clicks else False
    cls = "mode-btn active" if new_state else "mode-btn"
    style = {"marginLeft": "20px", "padding": "6px 14px",
             "borderRadius": "10px", "cursor": "pointer", "fontSize": "12px",
             "border": "1px solid var(--border-subtle)"}
    if new_state:
        style.update({"background": "var(--accent-green)",
                      "color": "#0a0a0a", "fontWeight": "600",
                      "borderColor": "transparent"})
    else:
        style.update({"background": "var(--bg-elevated)",
                      "color": "var(--text-secondary)"})
    return new_state, cls, style


# ---------------------------------------------------------------------------
# Callbacks: refresh button
# ---------------------------------------------------------------------------

@app.callback(
    Output("prices-status", "children"),
    Input("refresh-prices-btn", "n_clicks"),
    Input("refresh-trigger", "n_intervals"),
)
def refresh_prices(n_clicks, n_intervals):
    button_clicked = bool(n_clicks and n_clicks > 0 and
                          callback_context.triggered_id == "refresh-prices-btn")
    try:
        # Button click → wait for fresh prices (user has asked to wait).
        # Page-load / Interval → return cached immediately, refresh in bg.
        # Either way the UI renders instantly with whatever's in cache; the
        # button press just guarantees the cache is fresh before returning.
        prices = ds.refresh_live_prices(force=button_clicked,
                                        blocking=button_clicked)
        ds.get_value_panel(refresh=False)
    except Exception as e:
        return f"Price refresh error: {e!s}"
    ok = sum(1 for lp in prices.values() if lp.mid is not None)
    # Show Berlin time so the "updated at" is meaningful even when the user
    # runs the dashboard from a non-CET machine. Includes the offset
    # suffix (CET / CEST) so DST is unambiguous.
    from .performance import BERLIN_TZ
    now_berlin = datetime.now(BERLIN_TZ)
    return (f"Live prices: {ok}/{len(prices)} fresh · "
            f"updated {now_berlin.strftime('%H:%M:%S %Z')}")


# ---------------------------------------------------------------------------
# Callbacks: KPI cards + performance chart + metrics
# ---------------------------------------------------------------------------

def _window_slice(panel, period: str):
    ref = panel.dates[-1]
    start = window_start(ref, period, absolute_start=panel.dates[0])
    start = max(start, panel.dates[0])
    mask = (panel.dates >= start)
    return panel.value[mask], panel.external_flow[mask]


def _window_tax(panel, period: str) -> float:
    """Capital-gains tax paid in the window (positive number)."""
    ref = panel.dates[-1]
    start = window_start(ref, period, absolute_start=panel.dates[0])
    start = max(start, panel.dates[0])
    return float(panel.tax_paid[panel.dates >= start].sum())


@app.callback(
    Output("kpi-cards", "children"),
    Input("prices-status", "children"),     # triggers after price refresh
    Input("active-period", "data"),
)
def render_kpis(_status, period):
    panel = ds.get_value_panel()
    state = ds.load_state()

    # When the Scalable CLI is connected, its valuation.total IS the value
    # the user sees in their Scalable app, and INTRADAY from its performance
    # array IS the today's-change figure that app shows. Using them directly
    # eliminates the structural reconciliation gap we've been fighting.
    from . import scalable_api as _api
    api_value = _api.valuation_total() if _api.is_available() else None
    api_returns = (_api.absolute_return_by_period()
                   if _api.is_available() else None)

    if api_value is not None:
        today_val = float(api_value)
    else:
        today_val = ds.current_value()

    today_yest = float(panel.value.iloc[-2]) if len(panel.value) > 1 else today_val
    today_flow = float(panel.external_flow.iloc[-1])
    if api_returns is not None and api_returns.get("1D") is not None:
        # Scalable's own INTRADAY return — what the app's "today" card shows.
        todays_change = float(api_returns["1D"])
    else:
        todays_change = today_val - today_yest - today_flow
    todays_change_pct = (todays_change / today_yest) if today_yest else 0.0

    sub_val, sub_flow = _window_slice(panel, period)
    metrics = compute_metrics(sub_val, sub_flow)
    twr = metrics.get("twr", float("nan"))
    abs_pnl_gross = metrics.get("absolute_pnl", float("nan"))
    tax_window = _window_tax(panel, period)

    # When the Scalable CLI is connected, prefer its own per-window
    # simpleAbsoluteReturn for the gross PnL — it's the exact number the
    # Scalable app shows. Periods the API doesn't expose (e.g. 3Y) fall
    # back to our panel-based calculation.
    if api_returns is not None and api_returns.get(period) is not None:
        abs_pnl_gross = float(api_returns[period])

    abs_pnl_net = abs_pnl_gross - tax_window
    twr_net = (abs_pnl_net / (metrics["start_value"] + max(
        metrics["net_external_flow"], 0))) if metrics else float("nan")

    realized_total_gross = sum(r.realized_pnl for r in state.portfolio.realized)
    realized_total_net = realized_total_gross - float(panel.tax_paid.sum())

    # Unrealized P&L across all currently open positions (live mid - avg cost).
    unrealized_total = 0.0
    cost_basis_total = 0.0
    for isin, pos in state.portfolio.positions.items():
        lp = state.live_prices.get(isin)
        mid = lp.mid if (lp and lp.mid is not None) else pos.avg_cost
        unrealized_total += (mid - pos.avg_cost) * pos.shares
        cost_basis_total += pos.cost_basis
    unrealized_pct = (unrealized_total / cost_basis_total
                      if cost_basis_total else 0)

    def kpi(label, value_html, sub_html=""):
        return html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value_html, className="kpi-value"),
            html.Div(sub_html, className="kpi-delta"),
        ], className="kpi-card")

    cash_share = (state.portfolio.cash_balance / today_val * 100
                  if today_val else 0)

    return [
        kpi("Portfolio value",
            fmt_eur(today_val, 2),
            html.Span([
                html.Span(fmt_eur(todays_change, 2),
                          className=color_class(todays_change)),
                html.Span("  ·  ", className="muted"),
                html.Span(fmt_pct(todays_change_pct),
                          className=color_class(todays_change)),
                html.Span("  today", className="muted"),
            ])),
        kpi("Thereof cash",
            fmt_eur(state.portfolio.cash_balance, 2),
            html.Span(f"{cash_share:+.2f}% of portfolio", className="muted")),
        kpi(f"PnL ({period}) — gross €",
            html.Span(fmt_eur(abs_pnl_gross, 2),
                      className=color_class(abs_pnl_gross)),
            html.Span([
                "Net of tax: ",
                html.Span(fmt_eur(abs_pnl_net, 2),
                          className=color_class(abs_pnl_net)),
            ], className="muted")),
        kpi(f"Return ({period}) — TWR %",
            html.Span(fmt_pct(twr), className=color_class(twr)),
            html.Span(f"Tax paid: {fmt_eur(tax_window, 2)}",
                      className="muted")),
        kpi("Unrealized P&L (open positions)",
            html.Span(fmt_eur(unrealized_total, 2),
                      className=color_class(unrealized_total)),
            html.Span([
                html.Span(fmt_pct(unrealized_pct),
                          className=color_class(unrealized_total)),
                html.Span(f"  on €{cost_basis_total:,.0f} cost basis",
                          className="muted"),
            ])),
        kpi("Realized P&L (lifetime)",
            html.Span([
                html.Span(fmt_eur(realized_total_gross, 2),
                          className=color_class(realized_total_gross)),
                html.Span("  gross", className="muted",
                          style={"fontSize": "12px", "marginLeft": "4px"}),
            ]),
            html.Span([
                "Net of tax: ",
                html.Span(fmt_eur(realized_total_net, 2),
                          className=color_class(realized_total_net)),
            ])),
    ]


def _benchmark_window(value: pd.Series) -> pd.Series | None:
    """Return the benchmark price series sliced to ``value``'s window, or
    None if no benchmark data is available."""
    series = ds.get_benchmark_series()
    if series is None or series.empty or series.isna().all():
        return None
    sub = series.reindex(value.index).ffill().bfill()
    if sub.isna().all():
        return None
    return sub


def _benchmark_trace_eur(value, bench, start_val):
    """Cumulative euro gain assuming start_val was invested in the benchmark."""
    base = float(bench.iloc[0])
    if base <= 0:
        return None
    factor = bench / base
    return (factor - 1) * start_val


def _benchmark_trace_pct(bench):
    base = float(bench.iloc[0])
    if base <= 0:
        return None
    return (bench / base - 1) * 100


def _make_perf_figure(value, flow, mode: str, period: str,
                       benchmark_on: bool = False):
    if value.empty:
        return go.Figure()
    # V_0 is the end-of-day-0 value, which ALREADY includes any flow that
    # happened on day 0 (e.g. the very first deposit). Counting flow[0] in
    # cum_flow again would push the gain to −flow[0] at t=0. Strip it.
    start_val = value.iloc[0]
    cum_flow = flow.cumsum() - flow.iloc[0]

    if mode == "twr":
        series = (twr_series(value, flow) - 1) * 100
        hover_fmt = "%{y:+.2f}%"
        # ``.2~f`` trims trailing zeros: 10.00 → 10, 2.50 → 2.5, 2.55 → 2.55.
        y_axis = dict(ticksuffix="%", tickformat=".2~f")
    elif mode == "eur":
        series = value - start_val - cum_flow
        hover_fmt = "€%{y:+,.2f}"
        # No decimals on axis ticks for euro amounts; the hover still shows ".2f".
        y_axis = dict(tickprefix="€", tickformat=",.0f")
    else:
        cum_inflow = flow.clip(lower=0).cumsum() - max(flow.iloc[0], 0.0)
        denom = (start_val + cum_inflow).clip(lower=1e-9)
        series = (value - start_val - cum_flow) / denom * 100
        hover_fmt = "%{y:+.2f}%"
        y_axis = dict(ticksuffix="%", tickformat=".2~f")

    series = series.round(2)        # belt-and-braces: kill any extra decimals
    last = float(series.iloc[-1])
    line_color = COLOR_GREEN if last >= 0 else COLOR_RED

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=line_color, width=2, shape="spline", smoothing=0.45),
        fill="tozeroy",
        fillcolor=(COLOR_GREEN_FILL if last >= 0 else COLOR_RED_FILL),
        # Don't repeat the date — hovermode='x unified' prints it at the top.
        hovertemplate=hover_fmt + "<extra>Portfolio</extra>",
        name="Portfolio",
        # When the user hovers, plotly will fade in a marker at that point.
        marker=dict(size=8, color=line_color,
                    line=dict(color="#0d0d10", width=2)),
    ))

    if benchmark_on:
        bench = _benchmark_window(value)
        if bench is not None:
            if mode == "eur":
                bench_series = _benchmark_trace_eur(value, bench, start_val)
            else:
                bench_series = _benchmark_trace_pct(bench)
            if bench_series is not None:
                bench_series = bench_series.round(2)
                fig.add_trace(go.Scatter(
                    x=bench_series.index, y=bench_series.values, mode="lines",
                    line=dict(color="#a78bfa", width=2,
                              shape="spline", smoothing=0.45),
                    hovertemplate=hover_fmt + "<extra>" + BENCHMARK_NAME +
                                  "</extra>",
                    name=BENCHMARK_NAME,
                ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        # Generous room for tick labels — narrow margins were clipping the
        # leftmost euro / percent labels and the bottommost date.
        margin=dict(l=64, r=20, t=10, b=44),
        height=380,
        # Short snappy transition. The previous 400ms cubic-in-out produced
        # an aggressive "zoom" feel when the x-axis range changed a lot
        # (e.g. 1Y → 1M); a 180 ms linear tween is barely noticeable on
        # smaller changes and stops feeling janky on large ones.
        transition=dict(duration=180, easing="linear"),
        xaxis=dict(showgrid=False, zeroline=False, color=COLOR_GREY,
                   hoverformat="%d %b %Y", automargin=True,
                   tickfont=_CHART_FONT,
                   showspikes=True, spikemode="across", spikesnap="cursor",
                   spikecolor="rgba(40,235,207,0.4)",
                   spikethickness=1, spikedash="dot"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                   color=COLOR_GREY, automargin=True,
                   tickfont=_CHART_FONT, **y_axis),
        showlegend=benchmark_on,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=_CHART_FONT["family"],
                              color=COLOR_GREY, size=11)),
        hovermode="x unified",
        hoverlabel=_HOVER_LABEL,
    )
    return fig


@app.callback(
    Output("perf-chart", "figure"),
    Input("active-period", "data"),
    Input("return-mode", "data"),
    Input("benchmark-on", "data"),
    Input("prices-status", "children"),
)
def render_perf_chart(period, mode, benchmark_on, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period)
    return _make_perf_figure(value, flow, mode or "twr", period,
                              benchmark_on=bool(benchmark_on))


@app.callback(
    Output("value-chart", "figure"),
    Input("active-period", "data"),
    Input("benchmark-on", "data"),
    Input("prices-status", "children"),
)
def render_value_chart(period, benchmark_on, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period)
    if value.empty:
        return go.Figure()

    # Cumulative net deposits within the window (relative to start = 0).
    start_val_raw = float(value.iloc[0])
    net_deposits = (flow.cumsum() - flow.iloc[0]) + start_val_raw
    value = value.round(2)
    net_deposits = net_deposits.round(2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=value.index, y=value.values, mode="lines",
        line=dict(color=COLOR_GREEN, width=2, shape="spline", smoothing=0.45),
        fill="tozeroy", fillcolor=COLOR_GREEN_FILL,
        hovertemplate="€%{y:,.2f}<extra>Value</extra>",
        name="Portfolio value",
        marker=dict(size=8, color=COLOR_GREEN,
                    line=dict(color="#0d0d10", width=2)),
    ))
    fig.add_trace(go.Scatter(
        x=net_deposits.index, y=net_deposits.values, mode="lines",
        line=dict(color="rgba(255,255,255,0.45)", width=1.4, dash="dot"),
        hovertemplate="€%{y:,.2f}<extra>Capital invested</extra>",
        name="Capital invested",
    ))

    if benchmark_on:
        bench = _benchmark_window(value)
        if bench is not None and float(bench.iloc[0]) > 0:
            bench_eur = (bench / float(bench.iloc[0])) * start_val_raw
            bench_eur = bench_eur.round(2)
            fig.add_trace(go.Scatter(
                x=bench_eur.index, y=bench_eur.values, mode="lines",
                line=dict(color="#a78bfa", width=2,
                          shape="spline", smoothing=0.45),
                hovertemplate="€%{y:,.2f}<extra>" + BENCHMARK_NAME + "</extra>",
                name=BENCHMARK_NAME,
            ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=64, r=20, t=10, b=44),
        height=340,
        transition=dict(duration=180, easing="linear"),
        xaxis=dict(showgrid=False, zeroline=False, color=COLOR_GREY,
                   hoverformat="%d %b %Y", automargin=True,
                   tickfont=_CHART_FONT,
                   showspikes=True, spikemode="across", spikesnap="cursor",
                   spikecolor="rgba(40,235,207,0.4)",
                   spikethickness=1, spikedash="dot"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color=COLOR_GREY, tickprefix="€", tickformat=",.0f",
                   automargin=True, tickfont=_CHART_FONT),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=_CHART_FONT["family"],
                              color=COLOR_GREY, size=11)),
        hovermode="x unified",
        hoverlabel=_HOVER_LABEL,
    )
    return fig


@app.callback(
    Output("yearly-table", "children"),
    Input("prices-status", "children"),
)
def render_yearly_table(_status):
    state = ds.load_state()
    panel = ds.get_value_panel()

    realized_df = state.realized_df.copy()
    if not realized_df.empty:
        realized_df["year"] = pd.to_datetime(
            realized_df["sell_datetime"]).dt.year

    distributions = state.portfolio.distributions.copy()
    if not distributions.empty:
        distributions["year"] = pd.to_datetime(distributions["datetime"]).dt.year

    years = sorted(set(panel.dates.year))
    today = panel.dates[-1]
    rows = []
    for y in years:
        # Window from 30 Dec of previous year (to match Scalable's YTD anchor)
        # to 30 Dec of this year, clipped to available data.
        win_start = pd.Timestamp(y - 1, 12, 30)
        win_end = pd.Timestamp(y, 12, 31)
        win_start = max(win_start, panel.dates[0])
        win_end = min(win_end, today)
        if win_end <= win_start:
            continue

        mask = (panel.dates >= win_start) & (panel.dates <= win_end)
        v = panel.value[mask]
        f = panel.external_flow[mask]
        if v.empty:
            continue

        start_val = float(v.iloc[0])
        end_val = float(v.iloc[-1])
        net_flow = float(f.iloc[1:].sum())  # exclude the flow on win_start
        gross_pnl = end_val - start_val - net_flow
        tax_y = float(panel.tax_paid[mask].sum())
        net_pnl = gross_pnl - tax_y

        realized_y = (float(realized_df[realized_df["year"] == y][
            "realized_pnl"].sum()) if not realized_df.empty else 0.0)
        distributions_y = (float(distributions[distributions["year"] == y][
            "amount"].sum()) if not distributions.empty else 0.0)

        twr_y = (twr_series(v, f).iloc[-1] - 1) * 100 if len(v) > 1 else 0.0

        rows.append({
            "year": str(y) + ("  (YTD)" if win_end < pd.Timestamp(y, 12, 30)
                              else ""),
            "year_sort": y,
            "start_value": round(start_val, 2),
            "end_value": round(end_val, 2),
            "net_deposited": round(net_flow, 2),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "twr_pct": round(twr_y, 2),
            "realized_pnl": round(realized_y, 2),
            "distributions": round(distributions_y, 2),
            "tax_paid": round(tax_y, 2),
        })

    columns = [
        {"name": "Year", "id": "year"},
        {"name": "Start value (€)", "id": "start_value", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "End value (€)", "id": "end_value", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Net flows (€)", "id": "net_deposited", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Gross P&L (€)", "id": "gross_pnl", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "TWR (%)", "id": "twr_pct", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Realized (€)", "id": "realized_pnl", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Distributions (€)", "id": "distributions", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Tax paid (€)", "id": "tax_paid", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Net P&L (€)", "id": "net_pnl", "type": "numeric",
         "format": {"specifier": ",.2f"}},
    ]

    return dash_table.DataTable(
        columns=columns,
        data=rows,
        sort_action="native",
        sort_by=[{"column_id": "year", "direction": "desc"}],
        page_action="none",
        style_header=_DT_STYLE_HEADER,
        style_cell=_DT_STYLE_CELL_NOWRAP,
        style_table=_DT_STYLE_TABLE,
        style_data=_DT_STYLE_DATA,
        style_cell_conditional=[
            {"if": {"column_id": "year"}, "textAlign": "left",
             "fontWeight": "500"},
        ],
        style_data_conditional=(
            _pnl_heatmap_rules("gross_pnl", bands_pct=False)
            + _pnl_heatmap_rules("net_pnl", bands_pct=False)
            + _pnl_heatmap_rules("twr_pct", bands_pct=True)
            + _pnl_heatmap_rules("realized_pnl", bands_pct=False)
        ),
    )


@app.callback(
    Output("metrics-grid", "children"),
    Input("active-period", "data"),
    Input("prices-status", "children"),
)
def render_metrics(period, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period)
    if value.empty:
        return html.Div("No data for selected period.", className="muted")

    m = compute_metrics(value, flow)

    # Money-weighted return (XIRR) for the window: flows are -inflows + +outflows,
    # plus the final portfolio value as positive.
    flows_xirr = [(pd.Timestamp(value.index[0]), -float(value.iloc[0]))]
    for d, f in flow.items():
        if d == value.index[0]:
            continue
        if f != 0:
            flows_xirr.append((pd.Timestamp(d), -float(f)))
    flows_xirr.append((pd.Timestamp(value.index[-1]), float(value.iloc[-1])))
    xirr_val = xirr(flows_xirr)

    tax_window = _window_tax(panel, period)
    abs_pnl_net = m["absolute_pnl"] - tax_window

    # Two extra ratios beyond Sharpe/Sortino. Both are common and they live
    # cleanly off metrics we already have.
    #  - Calmar: annualised return per unit of max drawdown.
    #  - Profit factor: gross winning realized P&L / abs(gross losing
    #    realized P&L) for trades that closed inside this window.
    calmar = (m["annualised_twr"] / abs(m["max_drawdown"])
              if m["max_drawdown"] and m["annualised_twr"] is not None
              and not pd.isna(m["max_drawdown"]) else float("nan"))

    state = ds.load_state()
    win_start, win_end = value.index[0], value.index[-1]
    win_realized = [r for r in state.portfolio.realized
                    if win_start <= pd.Timestamp(r.sell_datetime.date())
                    <= win_end]
    gross_profit = sum(r.realized_pnl for r in win_realized
                       if r.realized_pnl > 0)
    gross_loss = sum(-r.realized_pnl for r in win_realized
                     if r.realized_pnl < 0)
    profit_factor = (gross_profit / gross_loss
                     if gross_loss > 0 else float("nan"))

    # Market metrics vs benchmark. Daily simple returns; for the portfolio
    # we strip the impact of external flows so deposits don't bias beta.
    bench = ds.get_benchmark_series()
    bench_window = bench.reindex(value.index).ffill() if bench is not None else None
    twr_idx = twr_series(value, flow)
    portfolio_daily = twr_idx.pct_change().dropna()
    if bench_window is not None and not bench_window.isna().all():
        bench_daily = bench_window.pct_change().dropna()
        mkt = compute_market_metrics(portfolio_daily, bench_daily)
        bench_total_return = (float(bench_window.iloc[-1])
                              / float(bench_window.iloc[0]) - 1)
    else:
        mkt = {"beta": float("nan"), "alpha": float("nan"),
               "r_squared": float("nan"), "tracking_error": float("nan"),
               "info_ratio": float("nan")}
        bench_total_return = float("nan")

    cells = [
        # --- 1) Relative returns (%) -----------------------------------
        ("Time-weighted return", fmt_pct(m["twr"]), color_class(m["twr"])),
        ("Simple return", fmt_pct(m["simple_return"]),
         color_class(m["simple_return"])),
        ("Annualised TWR", fmt_pct(m["annualised_twr"]),
         color_class(m["annualised_twr"])),
        ("Money-weighted (XIRR)", fmt_pct(xirr_val), color_class(xirr_val)),
        ("Benchmark return", fmt_pct(bench_total_return),
         color_class(bench_total_return)),
        ("Alpha (annualised)", fmt_pct(mkt["alpha"]),
         color_class(mkt["alpha"])),

        # --- 2) Core risk (sits up top per user spec) ------------------
        ("Volatility (ann.)", fmt_pct(m["volatility"]), "muted"),
        ("Max drawdown", fmt_pct(m["max_drawdown"]), "neg"),

        # --- 3) Absolute P&L (€) ---------------------------------------
        ("Absolute P&L (gross)", fmt_eur(m["absolute_pnl"]),
         color_class(m["absolute_pnl"])),
        ("Absolute P&L (net of tax)", fmt_eur(abs_pnl_net),
         color_class(abs_pnl_net)),
        ("Capital-gains tax paid", fmt_eur(tax_window), "neg"),
        ("Net external flow", fmt_eur(m["net_external_flow"]), "muted"),

        # --- 4) Ratios -------------------------------------------------
        ("Sharpe (rf=2%)", fmt_num(m["sharpe"], 2), "muted"),
        ("Sortino (rf=2%)", fmt_num(m["sortino"], 2), "muted"),
        ("Calmar (ann. / |MaxDD|)", fmt_num(calmar, 2), "muted"),
        ("Profit factor", fmt_num(profit_factor, 2), "muted"),
        (f"Beta (vs {BENCHMARK_NAME})", fmt_num(mkt["beta"], 2), "muted"),
        ("R²", fmt_num(mkt["r_squared"], 2), "muted"),
        ("Information ratio", fmt_num(mkt["info_ratio"], 2), "muted"),

        # --- 5) Tracking error (last) ----------------------------------
        ("Tracking error (ann.)", fmt_pct(mkt["tracking_error"]), "muted"),
    ]
    return html.Div([
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value " + cls,
                     style={"fontSize": "17px"}),
        ], className="kpi-card", style={"padding": "14px 16px"})
        for label, value, cls in cells
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
              "gap": "12px"})


# ---------------------------------------------------------------------------
# Positions table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DataTable helpers
# ---------------------------------------------------------------------------

_DT_STYLE_HEADER = {
    "backgroundColor": "var(--bg-card)",
    "color": "var(--text-secondary)",
    "fontWeight": "500",
    "textTransform": "uppercase",
    "fontSize": "11px",
    "letterSpacing": "0.04em",
    "borderBottom": "1px solid var(--border-subtle)",
    "borderTop": "none",
    "padding": "10px 12px",
}
_DT_STYLE_CELL = {
    "backgroundColor": "var(--bg-card)",
    "color": "var(--text-primary)",
    "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    "fontSize": "13px",
    "padding": "10px 12px",
    "border": "none",
    "borderBottom": "1px solid var(--border-subtle)",
}
_DT_STYLE_DATA = {"backgroundColor": "var(--bg-card)"}
_DT_STYLE_FILTER = {
    "backgroundColor": "var(--bg-elevated)",
    "color": "var(--text-primary)",
}
# Apply to every DataTable so wide tables scroll horizontally instead of
# clipping columns, and individual cells don't wrap.
_DT_STYLE_TABLE = {
    "overflowX": "auto",
    "minWidth": "100%",
}
_DT_STYLE_CELL_NOWRAP = {
    **_DT_STYLE_CELL,
    "whiteSpace": "nowrap",
    "minWidth": "85px",
    "maxWidth": "320px",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}


# Abbreviated labels for the transaction-type column. Original CSV strings
# are verbose; the abbreviations keep tables compact without losing meaning.
_TYPE_ABBREVIATIONS = {
    "Corporate action": "CA",
    "Reinvestment_Distribution": "Reinv. dist.",
    "Savings plan": "Saver",
    "Cash Transfer In": "Transfer In",
    "Cash Transfer Out": "Transfer Out",
    "Security transfer": "Sec. transfer",
    "Distribution": "Distrib.",
    "Withdrawal": "Withdraw",
    "Interest": "Interest",
    "Deposit": "Deposit",
    "Buy": "Buy",
    "Sell": "Sell",
    "Taxes": "Tax",
}


def abbrev_type(t: str | None) -> str:
    if t is None:
        return "—"
    return _TYPE_ABBREVIATIONS.get(t, t)


def _pnl_color_rules(*column_ids: str):
    rules = []
    for col in column_ids:
        rules.append({
            "if": {"filter_query": f"{{{col}}} > 0", "column_id": col},
            "color": COLOR_GREEN,
        })
        rules.append({
            "if": {"filter_query": f"{{{col}}} < 0", "column_id": col},
            "color": COLOR_RED,
        })
    return rules


def _pnl_heatmap_rules(column_id: str, bands_pct: bool = False):
    """Banded background fill so the biggest gainers and losers pop out.

    Order matters: later rules override earlier ones, so we layer
    light → medium → strong → strongest for each direction.
    """
    if bands_pct:
        pos_thresh = [0, 5, 20, 50]
        neg_thresh = [0, -5, -20, -50]
    else:
        pos_thresh = [0, 25, 250, 1000]
        neg_thresh = [0, -25, -250, -1000]

    green_shades = ["rgba(40,235,207,0.08)",
                    "rgba(40,235,207,0.18)",
                    "rgba(40,235,207,0.30)",
                    "rgba(40,235,207,0.45)"]
    red_shades = ["rgba(255,85,96,0.08)",
                  "rgba(255,85,96,0.18)",
                  "rgba(255,85,96,0.30)",
                  "rgba(255,85,96,0.45)"]

    rules = []
    for thresh, bg in zip(pos_thresh, green_shades):
        rules.append({
            "if": {"filter_query": f"{{{column_id}}} > {thresh}",
                   "column_id": column_id},
            "backgroundColor": bg,
            "color": COLOR_GREEN,
            "fontWeight": "500",
        })
    for thresh, bg in zip(neg_thresh, red_shades):
        rules.append({
            "if": {"filter_query": f"{{{column_id}}} < {thresh}",
                   "column_id": column_id},
            "backgroundColor": bg,
            "color": COLOR_RED,
            "fontWeight": "500",
        })
    return rules


@app.callback(
    Output("positions-table", "children"),
    Input("prices-status", "children"),
)
def render_positions(_status):
    state = ds.load_state()
    positions = state.portfolio.positions
    if not positions:
        return html.Div("No open positions.", className="muted")

    panel = ds.get_value_panel()
    today = panel.dates[-1]
    yesterday = panel.dates[-2] if len(panel.dates) > 1 else today
    px = panel.prices
    live = state.live_prices

    # Scalable's own FIFO cost basis per ISIN (sanity-check column). None
    # when the API isn't connected, in which case the column stays blank
    # and we don't trigger any drift highlighting.
    api_fifo = _scalable_api.fifo_price_by_isin()

    rows = []
    for isin, pos in positions.items():
        lp = live.get(isin)
        mid = lp.mid if (lp and lp.mid is not None) else None
        if mid is None and isin in px.columns:
            mid = float(px.loc[today, isin])
        if mid is None:
            mid = pos.avg_cost
        yest_price = (float(px.loc[yesterday, isin])
                      if isin in px.columns and yesterday in px.index else mid)
        market_value = pos.shares * mid
        unrealized = market_value - pos.cost_basis
        unrealized_pct = (unrealized / pos.cost_basis * 100) if pos.cost_basis else 0
        day_change = (mid - yest_price) * pos.shares
        day_change_pct = ((mid / yest_price - 1) * 100) if yest_price else 0
        # Per-share FIFO drift vs Scalable's reference.
        if api_fifo is not None:
            api_fp = api_fifo.get(isin)
            avg_drift_eur = (round((pos.avg_cost - api_fp) * pos.shares, 2)
                             if api_fp is not None else None)
            api_fifo_disp = round(api_fp, 4) if api_fp is not None else None
        else:
            avg_drift_eur = None
            api_fifo_disp = None
        rows.append({
            "isin": isin,
            "name": pos.description or isin,
            "shares": round(pos.shares, 4),
            "avg_cost": round(pos.avg_cost, 2),
            "api_fifo": api_fifo_disp,
            "fifo_drift": avg_drift_eur,
            "price": round(mid, 4),
            "value": round(market_value, 2),
            "weight": None,           # filled below
            "day_change": round(day_change, 2),
            "day_change_pct": round(day_change_pct, 2),
            "unrealized": round(unrealized, 2),
            "unrealized_pct": round(unrealized_pct, 2),
        })
    rows.sort(key=lambda r: r["value"], reverse=True)
    total_value = sum(r["value"] for r in rows)
    for r in rows:
        r["weight"] = round(r["value"] / total_value * 100, 2) if total_value else 0

    columns = [
        {"name": "Position", "id": "name"},
        {"name": "ISIN", "id": "isin"},
        {"name": "Shares", "id": "shares", "type": "numeric",
         "format": {"specifier": ",.4f"}},
        {"name": "Avg cost (€)", "id": "avg_cost", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Price (€)", "id": "price", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Value (€)", "id": "value", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Weight (%)", "id": "weight", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Today (€)", "id": "day_change", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Today (%)", "id": "day_change_pct", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Unrealized (€)", "id": "unrealized", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Unrealized (%)", "id": "unrealized_pct", "type": "numeric",
         "format": {"specifier": ",.2f"}},
    ]
    # Append Scalable-FIFO cross-check columns at the right end only when
    # the API is connected (so CSV-only users see no empty columns).
    if api_fifo is not None:
        columns += [
            {"name": "Scalable FIFO (€)", "id": "api_fifo", "type": "numeric",
             "format": {"specifier": ",.2f"}},
            {"name": "FIFO drift (€)", "id": "fifo_drift", "type": "numeric",
             "format": {"specifier": ",.2f"}},
        ]

    return html.Div([
        html.Div([
            "Total value (excl. cash): ",
            html.Span(fmt_eur(total_value), style={"fontWeight": "600"}),
            "  ·  Cash: ",
            html.Span(fmt_eur(state.portfolio.cash_balance),
                      style={"fontWeight": "600"}),
            "  ·  Click any row for the FIFO lot drilldown.",
        ], className="muted",
            style={"marginBottom": "12px", "fontSize": "12px"}),
        dash_table.DataTable(
            id="positions-datatable",
            columns=columns,
            data=rows,
            sort_action="native",
            sort_mode="single",
            sort_by=[{"column_id": "value", "direction": "desc"}],
            page_action="none",
            cell_selectable=True,
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_cell_conditional=[
                {"if": {"column_id": "name"}, "textAlign": "left",
                 "fontWeight": "500"},
                {"if": {"column_id": "isin"}, "textAlign": "left",
                 "color": "var(--text-tertiary)", "fontSize": "11px"},
            ],
            style_data_conditional=(
                _pnl_heatmap_rules("day_change", bands_pct=False)
                + _pnl_heatmap_rules("day_change_pct", bands_pct=True)
                + _pnl_heatmap_rules("unrealized", bands_pct=False)
                + _pnl_heatmap_rules("unrealized_pct", bands_pct=True)
                # Flag rows whose lifetime cost basis differs from
                # Scalable's by more than 5 EUR. Amber so it reads as
                # "investigate" rather than red/green PnL.
                + ([
                    {"if": {"filter_query": "{fifo_drift} > 5",
                            "column_id": "fifo_drift"},
                     "backgroundColor": "rgba(240,185,11,0.20)",
                     "color": "#f0b90b", "fontWeight": "500"},
                    {"if": {"filter_query": "{fifo_drift} < -5",
                            "column_id": "fifo_drift"},
                     "backgroundColor": "rgba(240,185,11,0.20)",
                     "color": "#f0b90b", "fontWeight": "500"},
                ] if api_fifo is not None else [])
            ),
        ),
    ])


@app.callback(
    Output("selected-isin", "data"),
    Input("positions-datatable", "active_cell"),
    State("positions-datatable", "data"),
    State("selected-isin", "data"),
    prevent_initial_call=True,
)
def select_position(active_cell, data, current):
    if not active_cell or not data:
        raise PreventUpdate
    row_idx = active_cell.get("row")
    if row_idx is None or row_idx >= len(data):
        raise PreventUpdate
    isin = data[row_idx].get("isin")
    return None if isin == current else isin


@app.callback(
    Output("position-detail", "children"),
    Input("selected-isin", "data"),
    Input("prices-status", "children"),
)
def render_position_detail(isin, _status):
    if not isin:
        return ""
    state = ds.load_state()
    pos: Position | None = state.portfolio.positions.get(isin)
    if pos is None:
        return ""
    lp = state.live_prices.get(isin)
    mid = lp.mid if (lp and lp.mid is not None) else pos.avg_cost

    lot_rows = []
    for lot in pos.lots:
        cur_value = lot.shares * mid
        cost = lot.shares * lot.unit_cost
        unr = cur_value - cost
        unr_pct = unr / cost if cost else 0
        lot_rows.append(html.Tr([
            html.Td(lot.open_datetime.strftime("%Y-%m-%d")),
            html.Td(lot.source_type),
            html.Td(fmt_num(lot.shares, 6)),
            html.Td(fmt_num(lot.original_shares, 6)),
            html.Td(fmt_eur(lot.unit_cost)),
            html.Td(fmt_eur(cost)),
            html.Td(fmt_eur(cur_value)),
            html.Td(html.Span(fmt_eur(unr), className=color_class(unr))),
            html.Td(html.Span(fmt_pct(unr_pct), className=color_class(unr_pct))),
        ]))

    realized_for_isin = [r for r in state.portfolio.realized if r.isin == isin]
    realized_total = sum(r.realized_pnl for r in realized_for_isin)

    distributions = state.portfolio.distributions
    if not distributions.empty:
        div_total = float(distributions[distributions["isin"] == isin]["amount"].sum())
    else:
        div_total = 0.0

    return html.Div([
        html.H4(f"{pos.description or isin}"),
        html.Div(
            f"ISIN {isin}  ·  Open lots: {len(pos.lots)}  ·  "
            f"Live mid: {fmt_eur(mid)}  ·  "
            f"Realized P&L (lifetime): {fmt_eur(realized_total)}  ·  "
            f"Distributions received: {fmt_eur(div_total)}",
            className="detail-meta"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Opened"), html.Th("Source"),
                html.Th("Remaining"), html.Th("Original"),
                html.Th("Unit cost"), html.Th("Cost basis"),
                html.Th("Market value"),
                html.Th("Unrealized"), html.Th("Unrealized %"),
            ])),
            html.Tbody(lot_rows),
        ], className="lot-table"),
    ], className="position-detail")


# ---------------------------------------------------------------------------
# Transactions tab
# ---------------------------------------------------------------------------

def _df_to_table(df: pd.DataFrame, columns: list[tuple[str, str, callable]]):
    """Render a DataFrame as a styled HTML table.

    ``columns`` is ``[(label, source_col, formatter), ...]`` where formatter
    takes the row value and returns a string or Dash component.
    """
    header = html.Tr([html.Th(label, style={"textAlign": "left"
                                            if i == 0 else "right"})
                      for i, (label, *_rest) in enumerate(columns)])
    body = []
    for _, row in df.iterrows():
        cells = []
        for i, (_, col, fmt) in enumerate(columns):
            val = row.get(col)
            cells.append(html.Td(fmt(val) if fmt else (val if val is not None else "—"),
                                 style={"textAlign": "left" if i == 0 else "right"}))
        body.append(html.Tr(cells))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      className="lot-table")


@app.callback(
    Output("realized-table", "children"),
    Input("prices-status", "children"),
)
def render_realized(_status):
    state = ds.load_state()
    df = state.realized_df.copy()
    if df.empty:
        return html.Div("No realized trades yet.", className="muted")
    df["return_pct"] = df.apply(
        lambda r: (r["realized_pnl"] / r["cost_basis"] * 100
                   if r["cost_basis"] else 0.0), axis=1)
    df["date"] = df["sell_datetime"].dt.strftime("%Y-%m-%d")
    records = [{
        "date": r.date,
        "description": r.description or "—",
        "isin": r.isin,
        "sell_type": abbrev_type(r.sell_type),
        "shares": round(float(r.shares), 4),
        "sell_price": round(float(r.sell_price), 4),
        "proceeds": round(float(r.proceeds), 2),
        "cost_basis": round(float(r.cost_basis), 2),
        "holding_days": int(r.holding_days),
        "realized_pnl": round(float(r.realized_pnl), 2),
        "return_pct": round(float(r.return_pct), 2),
    } for r in df.itertuples(index=False)]

    columns = [
        {"name": "Date", "id": "date"},
        {"name": "Position", "id": "description"},
        {"name": "ISIN", "id": "isin"},
        {"name": "Type", "id": "sell_type"},
        {"name": "Shares", "id": "shares", "type": "numeric",
         "format": {"specifier": ",.4f"}},
        {"name": "Sell price (€)", "id": "sell_price", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Proceeds (€)", "id": "proceeds", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Cost basis (€)", "id": "cost_basis", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Held (days)", "id": "holding_days", "type": "numeric"},
        {"name": "Realized P&L (€)", "id": "realized_pnl", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Return (%)", "id": "return_pct", "type": "numeric",
         "format": {"specifier": ",.2f"}},
    ]

    total = sum(r["realized_pnl"] for r in records)
    return html.Div([
        html.Div([
            "Lifetime realized P&L: ",
            html.Span(fmt_eur(total), className=color_class(total),
                      style={"fontWeight": "600"}),
            f"  ·  {len(records)} matched sells (sortable)",
        ], className="muted",
            style={"marginBottom": "12px", "fontSize": "12px"}),
        dash_table.DataTable(
            columns=columns,
            data=records,
            sort_action="native",
            sort_by=[{"column_id": "date", "direction": "desc"}],
            page_action="native",
            page_size=50,
            filter_action="native",
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_filter=_DT_STYLE_FILTER,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("date", "description", "isin", "sell_type")
            ],
            style_data_conditional=(
                _pnl_heatmap_rules("realized_pnl", bands_pct=False)
                + _pnl_heatmap_rules("return_pct", bands_pct=True)
            ),
        ),
    ])


# ---------------------------------------------------------------------------
# Best & Worst tab callbacks
# ---------------------------------------------------------------------------

_TRADE_COLUMNS = [
    {"name": "Date", "id": "date"},
    {"name": "Position", "id": "description"},
    {"name": "ISIN", "id": "isin"},
    {"name": "Type", "id": "sell_type"},
    {"name": "Shares", "id": "shares", "type": "numeric",
     "format": {"specifier": ",.4f"}},
    {"name": "Sell price (€)", "id": "sell_price", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Proceeds (€)", "id": "proceeds", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Cost basis (€)", "id": "cost_basis", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Held (days)", "id": "holding_days", "type": "numeric"},
    {"name": "Realized P&L (€)", "id": "realized_pnl", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Return (%)", "id": "return_pct", "type": "numeric",
     "format": {"specifier": ",.2f"}},
]


def _realized_records(state) -> list[dict]:
    """Re-shape realized trades for the Best/Worst tables."""
    df = state.realized_df.copy()
    if df.empty:
        return []
    df["return_pct"] = df.apply(
        lambda r: (r["realized_pnl"] / r["cost_basis"] * 100
                   if r["cost_basis"] else 0.0), axis=1)
    return [{
        "date": r.sell_datetime.strftime("%Y-%m-%d"),
        "description": r.description or "—",
        "isin": r.isin,
        "sell_type": abbrev_type(r.sell_type),
        "shares": round(float(r.shares), 4),
        "sell_price": round(float(r.sell_price), 4),
        "proceeds": round(float(r.proceeds), 2),
        "cost_basis": round(float(r.cost_basis), 2),
        "holding_days": int(r.holding_days),
        "realized_pnl": round(float(r.realized_pnl), 2),
        "return_pct": round(float(r.return_pct), 2),
    } for r in df.itertuples(index=False)]


def _trades_table(records: list[dict], header_note: str):
    return html.Div([
        html.Div(header_note, className="muted",
                 style={"marginBottom": "12px", "fontSize": "12px"}),
        dash_table.DataTable(
            columns=_TRADE_COLUMNS,
            data=records,
            sort_action="native",
            page_action="none",
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("date", "description", "isin", "sell_type")
            ],
            style_data_conditional=(
                _pnl_heatmap_rules("realized_pnl", bands_pct=False)
                + _pnl_heatmap_rules("return_pct", bands_pct=True)
            ),
        ),
    ])


@app.callback(
    Output("best-trades-table", "children"),
    Input("prices-status", "children"),
)
def render_best_trades(_status):
    recs = _realized_records(ds.load_state())
    if not recs:
        return html.Div("No realized trades yet.", className="muted")
    top = sorted(recs, key=lambda r: r["realized_pnl"], reverse=True)[:25]
    total = sum(r["realized_pnl"] for r in top)
    note = ["Top 25 by realized P&L  ·  combined: ",
            html.Span(fmt_eur(total), className="pos",
                      style={"fontWeight": "600"})]
    return _trades_table(top, note)


@app.callback(
    Output("worst-trades-table", "children"),
    Input("prices-status", "children"),
)
def render_worst_trades(_status):
    recs = _realized_records(ds.load_state())
    if not recs:
        return html.Div("No realized trades yet.", className="muted")
    bottom = sorted(recs, key=lambda r: r["realized_pnl"])[:25]
    total = sum(r["realized_pnl"] for r in bottom)
    note = ["Bottom 25 by realized P&L  ·  combined: ",
            html.Span(fmt_eur(total), className="neg",
                      style={"fontWeight": "600"})]
    return _trades_table(bottom, note)


_ISIN_COLUMNS = [
    {"name": "Position", "id": "description"},
    {"name": "ISIN", "id": "isin"},
    {"name": "Trades", "id": "trade_count", "type": "numeric"},
    {"name": "First", "id": "first_date"},
    {"name": "Last", "id": "last_date"},
    {"name": "Proceeds (€)", "id": "proceeds", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Cost basis (€)", "id": "cost_basis", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Realized P&L (€)", "id": "realized_pnl", "type": "numeric",
     "format": {"specifier": ",.2f"}},
    {"name": "Return (%)", "id": "return_pct", "type": "numeric",
     "format": {"specifier": ",.2f"}},
]


def _by_isin_records(state) -> list[dict]:
    """Aggregate realized trades per ISIN."""
    df = state.realized_df.copy()
    if df.empty:
        return []
    grouped = df.groupby("isin")
    rows = []
    for isin, sub in grouped:
        desc = (sub["description"].dropna().iloc[-1]
                if not sub["description"].dropna().empty else isin)
        cost = float(sub["cost_basis"].sum())
        rows.append({
            "description": desc or "—",
            "isin": isin,
            "trade_count": int(len(sub)),
            "first_date": sub["sell_datetime"].min().strftime("%Y-%m-%d"),
            "last_date": sub["sell_datetime"].max().strftime("%Y-%m-%d"),
            "proceeds": round(float(sub["proceeds"].sum()), 2),
            "cost_basis": round(cost, 2),
            "realized_pnl": round(float(sub["realized_pnl"].sum()), 2),
            "return_pct": round(float(sub["realized_pnl"].sum() / cost * 100)
                                if cost else 0.0, 2),
        })
    return rows


def _by_isin_table(records: list[dict], header_note):
    return html.Div([
        html.Div(header_note, className="muted",
                 style={"marginBottom": "12px", "fontSize": "12px"}),
        dash_table.DataTable(
            columns=_ISIN_COLUMNS,
            data=records,
            sort_action="native",
            page_action="none",
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("description", "isin", "first_date", "last_date")
            ],
            style_data_conditional=(
                _pnl_heatmap_rules("realized_pnl", bands_pct=False)
                + _pnl_heatmap_rules("return_pct", bands_pct=True)
            ),
        ),
    ])


@app.callback(
    Output("best-by-isin-table", "children"),
    Input("prices-status", "children"),
)
def render_best_by_isin(_status):
    recs = _by_isin_records(ds.load_state())
    if not recs:
        return html.Div("No realized trades yet.", className="muted")
    top = sorted(recs, key=lambda r: r["realized_pnl"], reverse=True)[:25]
    total = sum(r["realized_pnl"] for r in top)
    note = ["Top 25 positions by lifetime realized P&L  ·  combined: ",
            html.Span(fmt_eur(total), className="pos",
                      style={"fontWeight": "600"})]
    return _by_isin_table(top, note)


@app.callback(
    Output("worst-by-isin-table", "children"),
    Input("prices-status", "children"),
)
def render_worst_by_isin(_status):
    recs = _by_isin_records(ds.load_state())
    if not recs:
        return html.Div("No realized trades yet.", className="muted")
    bottom = sorted(recs, key=lambda r: r["realized_pnl"])[:25]
    total = sum(r["realized_pnl"] for r in bottom)
    note = ["Bottom 25 positions by lifetime realized P&L  ·  combined: ",
            html.Span(fmt_eur(total), className="neg",
                      style={"fontWeight": "600"})]
    return _by_isin_table(bottom, note)


@app.callback(
    Output("transactions-table", "children"),
    Input("prices-status", "children"),
)
def render_transactions(_status):
    state = ds.load_state()
    df = state.tx.raw.sort_values("datetime", ascending=False).copy()
    records = [{
        "date": r.datetime.strftime("%Y-%m-%d %H:%M"),
        "type": abbrev_type(r.type),
        "assetType": r.assetType,
        "description": r.description if pd.notna(r.description) else "—",
        "isin": r.isin if pd.notna(r.isin) else "—",
        "shares": round(float(r.shares), 4) if pd.notna(r.shares) and r.shares else None,
        "price": round(float(r.price), 4) if pd.notna(r.price) and r.price else None,
        "amount": round(float(r.amount), 2) if pd.notna(r.amount) else None,
        "fee": round(float(r.fee), 2) if pd.notna(r.fee) and r.fee else None,
        "tax": round(float(r.tax), 2) if pd.notna(r.tax) and r.tax else None,
    } for r in df.itertuples(index=False)]

    columns = [
        {"name": "Date", "id": "date"},
        {"name": "Type", "id": "type"},
        {"name": "Asset", "id": "assetType"},
        {"name": "Description", "id": "description"},
        {"name": "ISIN", "id": "isin"},
        {"name": "Shares", "id": "shares", "type": "numeric",
         "format": {"specifier": ",.4f"}},
        {"name": "Price (€)", "id": "price", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Amount (€)", "id": "amount", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Fee (€)", "id": "fee", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Tax (€)", "id": "tax", "type": "numeric",
         "format": {"specifier": ",.2f"}},
    ]

    return html.Div([
        html.Div(f"{len(records):,} executed transactions  ·  sortable and filterable",
                 className="muted",
                 style={"marginBottom": "12px", "fontSize": "12px"}),
        dash_table.DataTable(
            columns=columns,
            data=records,
            sort_action="native",
            sort_by=[{"column_id": "date", "direction": "desc"}],
            page_action="native",
            page_size=50,
            filter_action="native",
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_filter=_DT_STYLE_FILTER,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("date", "type", "assetType", "description", "isin")
            ],
            style_data_conditional=_pnl_color_rules("amount"),
        ),
    ])


@app.callback(
    Output("pending-orders-card", "children"),
    Input("prices-status", "children"),
)
def render_pending_orders(_status):
    """Only shows up when the Scalable API is connected AND at least one
    order is open/pending. Silent otherwise."""
    if not _scalable_api.is_available():
        return None
    pending = _scalable_api.pending_orders()
    if not pending:
        return None
    rows = [{
        "date": (p.get("last_event_datetime") or "")[:10],
        "side": p.get("side") or "—",
        "type": (p.get("security_transaction_type") or "").title()
                  or "Single",
        "quantity": p.get("quantity"),
        "description": p.get("description") or "—",
        "isin": p.get("isin") or "—",
        "limit_price": p.get("limit_price"),
        "amount": p.get("amount") or 0,
        "currency": p.get("currency") or "EUR",
        "status": (p.get("status") or "").title(),
    } for p in pending]
    columns = [
        {"name": "Date",       "id": "date"},
        {"name": "Side",       "id": "side"},
        {"name": "Type",       "id": "type"},
        {"name": "Quantity",   "id": "quantity", "type": "numeric",
         "format": {"specifier": ",.4f"}},
        {"name": "Position",   "id": "description"},
        {"name": "ISIN",       "id": "isin"},
        {"name": "Limit (€)",  "id": "limit_price", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Amount (€)", "id": "amount", "type": "numeric",
         "format": {"specifier": ",.2f"}},
        {"name": "Status",     "id": "status"},
    ]
    return html.Div([
        html.H3("Pending orders", className="card-title"),
        html.P("Open / unsettled orders straight from your Scalable "
               "account. Not yet in your CSV.", className="card-subtitle"),
        dash_table.DataTable(
            columns=columns, data=rows,
            sort_action="native",
            sort_by=[{"column_id": "date", "direction": "desc"}],
            page_action="none",
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("date", "side", "type", "description", "isin",
                          "status")
            ],
        ),
    ], className="card")


# ---------------------------------------------------------------------------
# Allocation tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("allocation-charts", "children"),
    Input("prices-status", "children"),
)
def render_allocation(_status):
    state = ds.load_state()
    positions = state.portfolio.positions
    if not positions:
        return html.Div("No positions.", className="muted")
    live = state.live_prices
    rows = []
    for isin, pos in positions.items():
        lp = live.get(isin)
        mid = (lp.mid if (lp and lp.mid is not None) else pos.avg_cost)
        rows.append({
            "isin": isin, "name": pos.description or isin,
            "value": pos.shares * mid,
            "region": _isin_region(isin),
        })
    df = pd.DataFrame(rows)

    pie_pos = go.Figure(data=[go.Pie(
        labels=df["name"], values=df["value"], hole=0.55,
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>",
        marker=dict(line=dict(color="#16181d", width=2)),
    )])
    pie_pos.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        showlegend=False,
        annotations=[dict(text="Positions",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(family=_CHART_FONT["family"],
                                    color=COLOR_GREY, size=14))],
        hoverlabel=_HOVER_LABEL,
    )

    region_agg = df.groupby("region")["value"].sum().reset_index()
    pie_region = go.Figure(data=[go.Pie(
        labels=region_agg["region"], values=region_agg["value"], hole=0.55,
        hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percent}<extra></extra>",
        marker=dict(line=dict(color="#16181d", width=2)),
    )])
    pie_region.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        showlegend=False,
        annotations=[dict(text="Region",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(family=_CHART_FONT["family"],
                                    color=COLOR_GREY, size=14))],
        hoverlabel=_HOVER_LABEL,
    )

    pies = [
        html.Div(dcc.Graph(figure=pie_pos, config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "260px"}),
    ]

    # When the Scalable CLI is connected, use its richer pre-computed
    # breakdowns (product type, equity sector, asset class, region)
    # instead of our ISIN-prefix country guess. Same pies fall back to
    # the country-guess region pie otherwise.
    api_breakdowns = _scalable_api.allocation_breakdowns()
    if api_breakdowns:
        for key in ("PRODUCT_TYPE", "ASSET_CLASS", "EQUITY_SECTOR", "REGION"):
            rows_api = api_breakdowns.get(key)
            if not rows_api:
                continue
            labels = [r["label"] for r in rows_api]
            values = [
                (r["value_eur"] if r["value_eur"] is not None
                 else r["weight"]) for r in rows_api
            ]
            pie = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.55,
                hovertemplate=("<b>%{label}</b><br>€%{value:,.2f}<br>"
                               "%{percent}<extra></extra>"),
                marker=dict(line=dict(color="#16181d", width=2)),
            )])
            pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=_CHART_FONT,
                margin=dict(l=10, r=10, t=10, b=10), height=320,
                showlegend=False,
                annotations=[dict(
                    text=_scalable_api.allocation_nice_name(key),
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(family=_CHART_FONT["family"],
                              color=COLOR_GREY, size=14))],
                hoverlabel=_HOVER_LABEL,
            )
            pies.append(html.Div(
                dcc.Graph(figure=pie, config={"displayModeBar": False}),
                style={"flex": "1", "minWidth": "260px"}))
    else:
        # API not available — keep the country-guess region pie.
        pies.append(html.Div(
            dcc.Graph(figure=pie_region, config={"displayModeBar": False}),
            style={"flex": "1", "minWidth": "260px"}))

    return html.Div(pies, style={"display": "flex", "gap": "20px",
                                  "flexWrap": "wrap"})


def _isin_region(isin: str) -> str:
    if not isin or len(isin) < 2:
        return "Other"
    cc = isin[:2]
    return {
        "US": "United States", "DE": "Germany", "FR": "France", "NL": "Netherlands",
        "IE": "Ireland", "LU": "Luxembourg", "GB": "United Kingdom",
        "CH": "Switzerland", "JP": "Japan", "CN": "China", "HK": "Hong Kong",
        "SG": "Singapore", "AU": "Australia", "CA": "Canada", "BE": "Belgium",
        "ES": "Spain", "IT": "Italy", "SE": "Sweden", "DK": "Denmark",
        "FI": "Finland", "NO": "Norway", "KY": "Cayman Islands",
        "BM": "Bermuda", "JE": "Jersey", "GG": "Guernsey",
        "AT": "Austria", "PT": "Portugal", "IL": "Israel",
        "BR": "Brazil", "MX": "Mexico", "KR": "South Korea",
        "TW": "Taiwan", "ZA": "South Africa", "IN": "India",
    }.get(cc, f"Other ({cc})")


@app.callback(
    Output("cash-chart", "figure"),
    Input("prices-status", "children"),
)
def render_cash_chart(_status):
    state = ds.load_state()
    cs = state.portfolio.cash_series
    if cs.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cs["datetime"], y=cs["cash"], mode="lines",
        line=dict(color=COLOR_LINE, width=2, shape="spline", smoothing=0.45),
        fill="tozeroy",
        fillcolor=COLOR_GREEN_FILL,
        hovertemplate="€%{y:,.2f}<extra></extra>",
        name="Cash",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=64, r=20, t=10, b=44), height=300,
        transition=dict(duration=180, easing="linear"),
        xaxis=dict(showgrid=False, color=COLOR_GREY,
                   hoverformat="%d %b %Y", automargin=True,
                   tickfont=_CHART_FONT,
                   showspikes=True, spikemode="across", spikesnap="cursor",
                   spikecolor="rgba(40,235,207,0.4)",
                   spikethickness=1, spikedash="dot"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color=COLOR_GREY, tickprefix="€", tickformat=",.0f",
                   automargin=True, tickfont=_CHART_FONT),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=_HOVER_LABEL,
    )
    return fig


@app.callback(
    Output("dividends-summary", "children"),
    Input("prices-status", "children"),
)
def render_dividends(_status):
    state = ds.load_state()
    distributions = state.portfolio.distributions
    if distributions.empty:
        return html.Div("No distributions or dividends on record.", className="muted")
    total = float(distributions["amount"].sum())
    by_isin = (distributions.groupby(["isin", "description"])["amount"]
               .sum().reset_index().sort_values("amount", ascending=False))
    records = [{
        "description": r.description or "—",
        "isin": r.isin,
        "amount": round(float(r.amount), 2),
    } for r in by_isin.itertuples(index=False)]
    return html.Div([
        html.Div(["Lifetime distribution income: ",
                  html.Span(fmt_eur(total), className="pos",
                            style={"fontWeight": "600"}),
                  f"  ·  Net interest (credit line / KKT): ",
                  html.Span(fmt_eur(state.portfolio.interest_total),
                            className=color_class(state.portfolio.interest_total),
                            style={"fontWeight": "600"}),
                  html.Span(
                      " — negative values are leverage cost from "
                      "Scalable's credit facility.",
                      className="muted", style={"fontStyle": "italic"})],
                 className="muted", style={"marginBottom": "12px",
                                           "fontSize": "12px"}),
        dash_table.DataTable(
            columns=[
                {"name": "Position", "id": "description"},
                {"name": "ISIN", "id": "isin"},
                {"name": "Distributions received (€)", "id": "amount",
                 "type": "numeric", "format": {"specifier": ",.2f"}},
            ],
            data=records,
            sort_action="native",
            sort_by=[{"column_id": "amount", "direction": "desc"}],
            page_action="none",
            style_header=_DT_STYLE_HEADER,
            style_cell=_DT_STYLE_CELL_NOWRAP,
            style_table=_DT_STYLE_TABLE,
            style_data=_DT_STYLE_DATA,
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("description", "isin")
            ],
            style_data_conditional=[
                {"if": {"column_id": "amount", "filter_query": "{amount} > 0"},
                 "color": COLOR_GREEN},
            ],
        ),
    ])


# ---------------------------------------------------------------------------
# Returns tab — data helpers
# ---------------------------------------------------------------------------

_RESAMPLE_MAP = {"D": None, "W": "W-FRI", "M": "ME"}
BENCH_COLOR = "#a78bfa"
BENCH_FILL = "rgba(167,139,250,0.18)"

def _gran_label(gran: str, what: str = "long") -> str:
    if what == "long":
        return {"D": "day", "W": "week", "M": "month"}.get(gran, "period")
    return {"D": "Daily", "W": "Weekly", "M": "Monthly"}.get(gran, "Period")


def _fmt_period_date(ts: pd.Timestamp, gran: str) -> str:
    """Format a period-end date in the way a human would label that bucket."""
    if gran == "M":
        return ts.strftime("%b %Y")          # "Jul 2025"
    if gran == "W":
        return f"Week of {ts.strftime('%d %b %Y')}"
    return ts.strftime("%d %b %Y")           # daily


def _portfolio_period_returns(value: pd.Series, flow: pd.Series,
                               gran: str, mode: str) -> pd.Series:
    """Period returns in either TWR % or EUR. ``value`` and ``flow`` are the
    panel series already sliced to the selected window."""
    if mode == "pct":
        idx = twr_series(value, flow)
        if gran == "D":
            r = idx.pct_change()
            r = r[idx.index.dayofweek < 5]
        else:
            r = idx.resample(_RESAMPLE_MAP[gran]).last().pct_change()
        return (r.dropna() * 100)
    daily = value.diff() - flow
    daily = daily.fillna(0)
    if gran == "D":
        return daily[value.index.dayofweek < 5]
    return daily.resample(_RESAMPLE_MAP[gran]).sum()


def _benchmark_period_returns(value: pd.Series, gran: str,
                               mode: str) -> pd.Series | None:
    bench = ds.get_benchmark_series()
    if bench is None or bench.empty or bench.isna().all():
        return None
    bench_window = bench.reindex(value.index).ffill().dropna()
    if bench_window.empty:
        return None
    if mode == "pct":
        if gran == "D":
            r = bench_window.pct_change()
            r = r[bench_window.index.dayofweek < 5]
        else:
            r = bench_window.resample(_RESAMPLE_MAP[gran]).last().pct_change()
        return (r.dropna() * 100)
    start_val = float(value.iloc[0])
    if start_val <= 0:
        return None
    ratio = bench_window / float(bench_window.iloc[0])
    eur_curve = ratio * start_val
    daily = eur_curve.diff().fillna(0)
    if gran == "D":
        return daily[bench_window.index.dayofweek < 5]
    return daily.resample(_RESAMPLE_MAP[gran]).sum()


def _max_streak(mask: pd.Series) -> int:
    best = cur = 0
    for b in mask.values:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ---------------------------------------------------------------------------
# Returns tab — chart callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("ret-bars-chart", "figure"),
    Input("ret-granularity", "data"),
    Input("ret-mode", "data"),
    Input("ret-benchmark-on", "data"),
    Input("active-period", "data"),
    Input("prices-status", "children"),
)
def render_ret_bars(gran, mode, bench_on, period, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period or DEFAULT_PERIOD)
    gran = gran or "M"
    mode = mode or "pct"
    rets = _portfolio_period_returns(value, flow, gran, mode).round(2)
    if rets.empty:
        return go.Figure()

    if mode == "pct":
        hover_fmt = "%{y:+.2f}%"
        y_axis = dict(ticksuffix="%", tickformat=".2~f")
    else:
        hover_fmt = "€%{y:+,.2f}"
        y_axis = dict(tickprefix="€", tickformat=",.0f")

    colors = [COLOR_GREEN if v >= 0 else COLOR_RED for v in rets.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rets.index, y=rets.values,
        marker_color=colors, marker_line_width=0,
        hovertemplate=hover_fmt + "<extra>Portfolio</extra>",
        name="Portfolio",
    ))
    if bench_on:
        b = _benchmark_period_returns(value, gran, mode)
        if b is not None and not b.empty:
            b = b.round(2)
            fig.add_trace(go.Scatter(
                x=b.index, y=b.values, mode="lines",
                line=dict(color=BENCH_COLOR, width=2,
                          shape="spline", smoothing=0.45),
                hovertemplate=hover_fmt + "<extra>" + BENCHMARK_NAME + "</extra>",
                name=BENCHMARK_NAME,
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=64, r=20, t=10, b=44),
        height=380,
        transition=dict(duration=180, easing="linear"),
        xaxis=dict(showgrid=False, zeroline=False, color="#9aa0aa",
                   hoverformat=("%b %Y" if gran == "M"
                                else "%d %b %Y"),
                   automargin=True, tickfont=_CHART_FONT),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                   color="#9aa0aa", automargin=True,
                   tickfont=_CHART_FONT, **y_axis),
        showlegend=bench_on,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=_CHART_FONT["family"],
                              color="#9aa0aa", size=11)),
        hovermode="x unified",
        hoverlabel=_HOVER_LABEL,
        bargap=0.1,
    )
    return fig


@app.callback(
    Output("ret-hist-chart", "figure"),
    Input("ret-granularity", "data"),
    Input("ret-mode", "data"),
    Input("ret-benchmark-on", "data"),
    Input("active-period", "data"),
    Input("prices-status", "children"),
)
def render_ret_histogram(gran, mode, bench_on, period, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period or DEFAULT_PERIOD)
    gran = gran or "M"
    mode = mode or "pct"
    rets = _portfolio_period_returns(value, flow, gran, mode).round(2)
    if rets.empty:
        return go.Figure()
    is_pct = (mode == "pct")
    fmt = "%{x:+.2f}%" if is_pct else "€%{x:+,.2f}"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets.values, nbinsx=40,
        marker=dict(color=COLOR_GREEN_FILL,
                    line=dict(color=COLOR_GREEN, width=1)),
        hovertemplate=fmt + "<br>count %{y}<extra>Portfolio</extra>",
        name="Portfolio",
    ))
    if bench_on:
        b = _benchmark_period_returns(value, gran, mode)
        if b is not None and not b.empty:
            fig.add_trace(go.Histogram(
                x=b.values, nbinsx=40,
                marker=dict(color=BENCH_FILL,
                            line=dict(color=BENCH_COLOR, width=1)),
                hovertemplate=fmt + "<br>count %{y}<extra>" + BENCHMARK_NAME + "</extra>",
                name=BENCHMARK_NAME,
                opacity=0.65,
            ))

    mean = float(rets.mean())
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.25)", width=1,
                                 dash="dot"))
    mean_label = (f"mean {mean:+.2f}%" if is_pct
                  else f"mean €{mean:+,.2f}")
    fig.add_vline(x=mean, line=dict(color=COLOR_GREEN, width=1.5, dash="dash"),
                  annotation_text=mean_label,
                  annotation_position="top right",
                  annotation_font=dict(family=_CHART_FONT["family"],
                                       color=COLOR_GREEN, size=11))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=64, r=20, t=10, b=44),
        height=340,
        barmode="overlay",
        xaxis=dict(showgrid=False, color="#9aa0aa", automargin=True,
                   tickfont=_CHART_FONT,
                   **({"ticksuffix": "%", "tickformat": ".2~f"} if is_pct
                      else {"tickprefix": "€", "tickformat": ",.0f"})),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color="#9aa0aa", automargin=True,
                   tickfont=_CHART_FONT, title=dict(text="Count",
                                                    font=_CHART_FONT)),
        showlegend=bench_on,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=_CHART_FONT["family"],
                              color="#9aa0aa", size=11)),
        hoverlabel=_HOVER_LABEL,
    )
    return fig


@app.callback(
    Output("ret-heatmap-chart", "figure"),
    Input("active-period", "data"),
    Input("prices-status", "children"),
)
def render_ret_heatmap(period, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period or DEFAULT_PERIOD)
    idx = twr_series(value, flow)
    monthly = (idx.resample("ME").last().pct_change().dropna() * 100)
    if monthly.empty:
        return go.Figure()

    df = pd.DataFrame({"r": monthly.values, "date": monthly.index})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.pivot(index="year", columns="month", values="r")
    pivot = pivot.reindex(columns=range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    text = [[("" if pd.isna(v) else f"{v:+.2f}%") for v in row]
            for row in pivot.values]

    vmax = float(max(abs(monthly.min()), abs(monthly.max())) or 1)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=[str(y) for y in pivot.index.tolist()],
        zmin=-vmax, zmax=vmax,
        colorscale=[[0.0, COLOR_RED], [0.5, "#16181d"], [1.0, COLOR_GREEN]],
        hovertemplate="%{y} %{x}<br>%{z:+.2f}%<extra></extra>",
        text=text, texttemplate="%{text}",
        textfont=dict(family=_CHART_FONT["family"], size=11,
                      color="#f4f4f6"),
        colorbar=dict(thickness=10, len=0.7, ticksuffix="%",
                      tickformat=".2~f",
                      tickfont=dict(family=_CHART_FONT["family"],
                                    color="#9aa0aa", size=10)),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=64, r=20, t=10, b=20),
        height=max(220, 40 * len(pivot.index)),
        xaxis=dict(color="#9aa0aa", side="top", tickfont=_CHART_FONT),
        yaxis=dict(color="#9aa0aa", autorange="reversed",
                   tickfont=_CHART_FONT),
        hoverlabel=_HOVER_LABEL,
    )
    return fig


@app.callback(
    Output("ret-drawdown-chart", "figure"),
    Input("active-period", "data"),
    Input("prices-status", "children"),
)
def render_ret_drawdown(period, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period or DEFAULT_PERIOD)
    idx = twr_series(value, flow)
    running_max = idx.cummax()
    dd = ((idx / running_max - 1) * 100).round(2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines",
        line=dict(color=COLOR_RED, width=2, shape="spline", smoothing=0.45),
        fill="tozeroy", fillcolor=COLOR_RED_FILL,
        hovertemplate="%{y:+.2f}%<extra>Drawdown</extra>",
        name="Drawdown",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=_CHART_FONT,
        margin=dict(l=64, r=20, t=10, b=44), height=320,
        transition=dict(duration=180, easing="linear"),
        xaxis=dict(showgrid=False, zeroline=False, color="#9aa0aa",
                   hoverformat="%d %b %Y", automargin=True,
                   tickfont=_CHART_FONT,
                   showspikes=True, spikemode="across", spikesnap="cursor",
                   spikecolor="rgba(255,85,96,0.4)",
                   spikethickness=1, spikedash="dot"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                   color="#9aa0aa", automargin=True, tickfont=_CHART_FONT,
                   ticksuffix="%", tickformat=".2~f"),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=_HOVER_LABEL,
    )
    return fig


@app.callback(
    Output("ret-stats-grid", "children"),
    Input("ret-granularity", "data"),
    Input("ret-mode", "data"),
    Input("active-period", "data"),
    Input("prices-status", "children"),
)
def render_ret_stats(gran, mode, period, _status):
    panel = ds.get_value_panel()
    value, flow = _window_slice(panel, period or DEFAULT_PERIOD)
    gran = gran or "M"
    pct = _portfolio_period_returns(value, flow, gran, "pct")
    if pct.empty:
        return html.Div("No data for the selected period.", className="muted")

    n = len(pct)
    pos = int((pct > 0).sum())
    hit = pos / n if n else 0
    best_idx = pct.idxmax()
    worst_idx = pct.idxmin()
    winners = pct[pct > 0]
    losers = pct[pct < 0]
    streak_pos = _max_streak(pct > 0)
    streak_neg = _max_streak(pct < 0)

    period_word = _gran_label(gran, "long")
    plural = period_word + "s"

    def fp(v): return fmt_pct(v / 100)

    cells = [
        (f"Periods ({plural})", f"{n}", "muted"),
        ("Hit rate", fmt_pct(hit), "pos" if hit > 0.5 else "muted"),
        (f"Best {period_word}",
         f"{fp(pct.max())}  ·  {_fmt_period_date(best_idx, gran)}", "pos"),
        (f"Worst {period_word}",
         f"{fp(pct.min())}  ·  {_fmt_period_date(worst_idx, gran)}", "neg"),
        ("Mean return", fp(pct.mean()), color_class(pct.mean())),
        ("Median return", fp(pct.median()), color_class(pct.median())),
        (f"Std dev per {period_word}", fp(pct.std()), "muted"),
        (f"Avg winning {period_word}",
         fp(winners.mean()) if not winners.empty else "—", "pos"),
        (f"Avg losing {period_word}",
         fp(losers.mean()) if not losers.empty else "—", "neg"),
        ("Longest pos. streak", f"{streak_pos} {plural}", "pos"),
        ("Longest neg. streak", f"{streak_neg} {plural}", "neg"),
        ("Skew", fmt_num(float(pct.skew()), 2) if len(pct) > 2 else "—",
         "muted"),
    ]
    return html.Div([
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value " + cls,
                     style={"fontSize": "15px"}),
        ], className="kpi-card", style={"padding": "12px 14px"})
        for label, value, cls in cells
    ], style={"display": "grid",
              "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    print("[startup] warming live prices (this can take a minute on first run)...")
    try:
        ds.refresh_live_prices()
        ds.get_value_panel()
        # Background download of benchmark history; cached after first run.
        ds.get_benchmark_series()
    except Exception as e:
        print(f"[startup] warning: warmup failed: {e}")
    print("[startup] launching Dash on http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
