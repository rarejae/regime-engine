"""Experiment visualizer — Streamlit dashboard.

Plug-and-play: drop a package under viz/packages/<id>/ matching viz/schema.py
and it appears in the sidebar. No app changes required for new experiments.

Run:
  .venv/bin/streamlit run viz/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Repo root on path so `viz.*` imports work under `streamlit run viz/app.py`
# (Streamlit Cloud and local).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from viz.metrics import (
    CRISIS_WINDOWS,
    cagr,
    crisis_drawdowns,
    dca_path,
    dca_terminal,
    drawdown_curve,
    equity_curve,
    metrics_from_daily,
)
from viz.schema import REQUIRED_FILES
from live.tax import after_tax_monthly_returns, effective_gain_rate

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"


# ── Package I/O ──────────────────────────────────────────────────────────────

def list_packages() -> list[Path]:
    if not PACKAGES.exists():
        return []
    out = []
    for p in sorted(PACKAGES.iterdir()):
        if p.is_dir() and all((p / f).exists() for f in REQUIRED_FILES):
            out.append(p)
    return out


@st.cache_data(show_spinner=False)
def load_package(path_str: str) -> dict:
    path = Path(path_str)
    meta = json.loads((path / "meta.json").read_text())
    daily = pd.read_parquet(path / "daily_returns.parquet")
    daily.index = pd.to_datetime(daily.index)
    monthly = pd.read_parquet(path / "monthly_returns.parquet")
    monthly.index = pd.to_datetime(monthly.index)
    state = None
    if (path / "monthly_state.parquet").exists():
        state = pd.read_parquet(path / "monthly_state.parquet")
        state.index = pd.to_datetime(state.index)
    cb = None
    if (path / "cb_events.csv").exists():
        cb = pd.read_csv(path / "cb_events.csv", parse_dates=["date"])
    return {"meta": meta, "daily": daily, "monthly": monthly, "state": state, "cb": cb}


def strategy_map(meta: dict) -> dict[str, dict]:
    return {s["id"]: s for s in meta["strategies"]}


def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{digits}%}"


def fmt_usd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:,.0f}"


def fmt_num(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


# ── Charts ───────────────────────────────────────────────────────────────────

def chart_equity(daily: pd.DataFrame, selected: list[str], smap: dict, log_scale: bool):
    fig = go.Figure()
    for sid in selected:
        eq = equity_curve(daily[sid])
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=smap[sid]["name"],
            line=dict(color=smap[sid].get("color"), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title="Growth of $1 (lump sum)",
        yaxis_title="Terminal wealth ($)",
        yaxis_type="log" if log_scale else "linear",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        hovermode="x unified",
    )
    return fig


def chart_drawdown(daily: pd.DataFrame, selected: list[str], smap: dict):
    fig = go.Figure()
    for sid in selected:
        dd = drawdown_curve(daily[sid])
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=smap[sid]["name"],
            line=dict(color=smap[sid].get("color"), width=1.5),
            fill="tozeroy" if sid == selected[0] else None,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
        ))
    fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=320,
        hovermode="x unified",
    )
    return fig


def chart_dca(monthly: pd.DataFrame, selected: list[str], smap: dict,
              start: float, contrib: float):
    fig = go.Figure()
    for sid in selected:
        path = dca_path(monthly[sid], start=start, contrib=contrib)
        fig.add_trace(go.Scatter(
            x=path.index, y=path.values, name=smap[sid]["name"],
            line=dict(color=smap[sid].get("color"), width=2),
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.0f}<extra></extra>",
        ))
    if len(monthly) and selected:
        idx = monthly.index
        invested = [start + contrib * (i + 1) for i in range(len(idx))]
        fig.add_trace(go.Scatter(
            x=idx, y=invested, name="Contributions only",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.0f}<extra></extra>",
        ))
    title = f"Wealth path — ${start:,.0f} start"
    title += f" + ${contrib:,.0f}/mo" if contrib else " (lump sum)"
    fig.update_layout(
        title=title,
        yaxis_title="Portfolio value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        hovermode="x unified",
    )
    return fig


def chart_annual(daily: pd.DataFrame, selected: list[str], smap: dict):
    years = sorted(set(daily.index.year))
    fig = go.Figure()
    for sid in selected:
        vals = []
        for y in years:
            sp = daily[sid][daily.index.year == y]
            vals.append((1 + sp).prod() - 1 if len(sp) else np.nan)
        fig.add_trace(go.Bar(
            x=years, y=vals, name=smap[sid]["name"],
            marker_color=smap[sid].get("color"),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        ))
    fig.update_layout(
        title="Annual returns",
        yaxis_tickformat=".0%",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=360,
    )
    return fig


def chart_state(state: pd.DataFrame):
    if state is None or state.empty or "eff_equity" not in state.columns:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08,
                        subplot_titles=("Effective equity", "Faber scores"))
    fig.add_trace(go.Scatter(
        x=state.index, y=state["eff_equity"], name="Eff equity",
        line=dict(color="#2563eb", width=1.5),
        fill="tozeroy",
        hovertemplate="%{x|%Y-%m}<br>%{y:.0%}<extra></extra>",
    ), row=1, col=1)
    for col, color, name in [
        ("qqq_sc", "#0ea5e9", "QQQ"),
        ("ivv_sc", "#64748b", "IVV"),
        ("iau_sc", "#f59e0b", "IAU"),
    ]:
        if col in state.columns:
            fig.add_trace(go.Scatter(
                x=state.index, y=state[col], name=name,
                line=dict(color=color, width=1.5),
                mode="lines",
            ), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(dtick=1, range=[-0.2, 3.2], row=2, col=1)
    fig.update_layout(height=480, margin=dict(l=40, r=20, t=40, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      hovermode="x unified")
    return fig


# ── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TAA Experiment Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("TAA Experiment Visualizer")
st.caption("Backtest comparison for regime-engine experiments. Drop a package in `viz/packages/` to add a run.")

packages = list_packages()
if not packages:
    st.error(
        "No experiment packages found. Generate one first:\n\n"
        "```bash\n.venv/bin/python viz/export_v19d_marketstack.py\n```"
    )
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Experiment")
    pkg_labels = {}
    for p in packages:
        meta_preview = json.loads((p / "meta.json").read_text())
        pkg_labels[f"{meta_preview.get('title', p.name)}"] = str(p)
    choice = st.selectbox("Package", list(pkg_labels.keys()))
    pkg = load_package(pkg_labels[choice])
    meta = pkg["meta"]
    daily_full = pkg["daily"]
    monthly_full = pkg["monthly"]
    smap = strategy_map(meta)

    st.markdown(f"**Window:** {meta['date_start']} → {meta['date_end']}")
    if meta.get("data_notes"):
        st.caption(meta["data_notes"])

    st.divider()
    st.header("Date range")
    min_d, max_d = daily_full.index.min().date(), daily_full.index.max().date()
    start_d = st.date_input("Start", value=min_d, min_value=min_d, max_value=max_d)
    end_d = st.date_input("End", value=max_d, min_value=min_d, max_value=max_d)
    if start_d >= end_d:
        st.error("Start must be before end")
        st.stop()

    st.divider()
    st.header("Strategies")
    default_ids = [s["id"] for s in meta["strategies"]]
    selected = st.multiselect(
        "Compare",
        options=default_ids,
        default=default_ids,
        format_func=lambda i: smap[i]["name"],
    )
    if not selected:
        st.warning("Select at least one strategy")
        st.stop()

    st.divider()
    st.header("Contributions")
    start_cap = st.number_input(
        "Starting principal ($)",
        min_value=0.0,
        value=10_000.0,
        step=1_000.0,
        format="%.0f",
        help="Initial lump-sum invested at the start of the selected window.",
    )
    monthly_contrib = st.number_input(
        "Monthly contribution ($)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.0f",
        help="Added at each month-end after that month's return. Set to 0 for lump sum only.",
    )
    log_scale = st.checkbox("Log scale equity curve", value=True)

    st.divider()
    st.header("Tax drag (taxable)")
    apply_tax = st.toggle("Show after-tax estimate", value=False)
    ordinary = st.slider("Federal ordinary rate", 0.0, 0.45, 0.32, 0.01)
    ltcg_rate = st.slider("Federal LTCG rate", 0.0, 0.25, 0.15, 0.01)
    state_tax = st.slider("State tax rate", 0.0, 0.15, 0.05, 0.005)
    stcg_frac = st.slider("Share of gains taxed as short-term", 0.0, 1.0, 0.80, 0.05)
    st.caption("Sensitivity model — not tax advice. See live/tax.py.")

# Slice data
daily = daily_full.loc[str(start_d):str(end_d), selected].dropna(how="all")
monthly = monthly_full.loc[str(start_d):str(end_d), selected].dropna(how="all")
state = pkg["state"]
if state is not None:
    state = state.loc[str(start_d):str(end_d)]

# ── Metrics table ────────────────────────────────────────────────────────────

rows = []
for sid in selected:
    m = metrics_from_daily(daily[sid].dropna())
    wealth = dca_terminal(monthly[sid].dropna(), start_cap, monthly_contrib)
    rows.append({
        "Strategy": smap[sid]["name"],
        "CAGR": m["CAGR"],
        "Vol": m["Vol"],
        "Sharpe": m["Sharpe"],
        "Sortino": m["Sortino"],
        "MaxDD": m["MaxDD"],
        "Calmar": m["Calmar"],
        "Terminal $1": m["Terminal_$1"],
        "Wealth terminal": wealth,
    })
metrics_df = pd.DataFrame(rows).set_index("Strategy")

st.subheader("Performance")
primary = meta.get("primary_strategy")
if primary in selected:
    pm = metrics_df.loc[smap[primary]["name"]]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", fmt_pct(pm["CAGR"]))
    c2.metric("Sharpe", fmt_num(pm["Sharpe"]))
    c3.metric("Max DD", fmt_pct(pm["MaxDD"], 1))
    c4.metric("Terminal $1", fmt_usd(pm["Terminal $1"]))
    c5.metric("Wealth terminal", fmt_usd(pm["Wealth terminal"]))

display = metrics_df.copy()
display["CAGR"] = display["CAGR"].map(lambda x: fmt_pct(x))
display["Vol"] = display["Vol"].map(lambda x: fmt_pct(x))
display["Sharpe"] = display["Sharpe"].map(lambda x: fmt_num(x))
display["Sortino"] = display["Sortino"].map(lambda x: fmt_num(x))
display["MaxDD"] = display["MaxDD"].map(lambda x: fmt_pct(x, 1))
display["Calmar"] = display["Calmar"].map(lambda x: fmt_num(x, 2))
display["Terminal $1"] = display["Terminal $1"].map(fmt_usd)
display["Wealth terminal"] = display["Wealth terminal"].map(fmt_usd)
st.dataframe(display, use_container_width=True)

if apply_tax:
    st.subheader("After-tax sensitivity (taxable account)")
    blend = effective_gain_rate(ordinary, ltcg_rate, state_tax, stcg_frac)
    st.caption(f"Blended marginal rate on realized gains ≈ **{blend:.1%}** "
               f"(STCG frac {stcg_frac:.0%}). Haircuts positive months only.")
    tax_rows = []
    at_monthly = {}
    for sid in selected:
        at = after_tax_monthly_returns(
            monthly[sid],
            state if sid in ("v19d", "sfev3") else None,
            ordinary, ltcg_rate, state_tax, stcg_frac,
        )
        at_monthly[sid] = at
        pre_wealth = dca_terminal(monthly[sid].dropna(), start_cap, monthly_contrib)
        at_wealth = dca_terminal(at, start_cap, monthly_contrib)
        pre_term = float((1 + monthly[sid].dropna()).prod())
        at_term = float((1 + at).prod()) if len(at) else float("nan")
        tax_rows.append({
            "Strategy": smap[sid]["name"],
            "CAGR pre (daily)": fmt_pct(metrics_from_daily(daily[sid].dropna())["CAGR"]),
            "CAGR after-tax approx": fmt_pct(cagr(at, "monthly")),
            "Terminal $1 pre": fmt_usd(pre_term),
            "Terminal $1 after-tax": fmt_usd(at_term),
            "Wealth pre": fmt_usd(pre_wealth),
            "Wealth after-tax": fmt_usd(at_wealth),
            "Drag on terminal $1": fmt_pct((at_term / pre_term - 1) if pre_term else float("nan"), 1),
        })
    st.dataframe(pd.DataFrame(tax_rows).set_index("Strategy"), use_container_width=True)

    fig_tax = go.Figure()
    for sid in selected:
        pre_path = dca_path(monthly[sid], start_cap, monthly_contrib)
        at_path = dca_path(at_monthly[sid], start_cap, monthly_contrib)
        fig_tax.add_trace(go.Scatter(
            x=pre_path.index, y=pre_path.values, name=f"{smap[sid]['name']} pre-tax",
            line=dict(color=smap[sid].get("color"), width=2),
        ))
        fig_tax.add_trace(go.Scatter(
            x=at_path.index, y=at_path.values, name=f"{smap[sid]['name']} after-tax",
            line=dict(color=smap[sid].get("color"), width=2, dash="dash"),
        ))
    fig_tax.update_layout(
        title="Wealth path — pre-tax vs after-tax estimate",
        yaxis_title="Portfolio value ($)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_tax, use_container_width=True)

# Crisis table
st.subheader("Crisis drawdowns")
crisis_rows = []
for sid in selected:
    cds = crisis_drawdowns(daily[sid].dropna(), CRISIS_WINDOWS)
    crisis_rows.append({"Strategy": smap[sid]["name"], **{k: fmt_pct(v, 1) for k, v in cds.items()}})
st.dataframe(pd.DataFrame(crisis_rows).set_index("Strategy"), use_container_width=True)

# Charts
st.plotly_chart(chart_equity(daily, selected, smap, log_scale), use_container_width=True)
st.plotly_chart(chart_drawdown(daily, selected, smap), use_container_width=True)
st.plotly_chart(
    chart_dca(monthly, selected, smap, start_cap, monthly_contrib),
    use_container_width=True,
)
st.plotly_chart(chart_annual(daily, selected, smap), use_container_width=True)

# State / CB for primary strategy
if state is not None and primary in selected:
    st.subheader(f"{smap[primary]['name']} — allocation state")
    fig_state = chart_state(state)
    if fig_state:
        st.plotly_chart(fig_state, use_container_width=True)

if pkg["cb"] is not None and len(pkg["cb"]):
    cb = pkg["cb"]
    cb = cb[(cb["date"] >= pd.Timestamp(start_d)) & (cb["date"] <= pd.Timestamp(end_d))]
    with st.expander(f"Circuit breaker events ({len(cb)})"):
        st.dataframe(cb, use_container_width=True, hide_index=True)

with st.expander("About this package"):
    st.json({k: meta[k] for k in meta if k != "strategies"})
    st.markdown(
        "**Metrics:** CAGR / Vol / Sharpe / Sortino / Calmar annualized from **daily** "
        "returns (252 trading days). **Terminal $1** is lump-sum growth of $1. "
        "**Wealth terminal** uses the sidebar starting principal and monthly contribution "
        "(contribution applied after each month's return)."
    )
