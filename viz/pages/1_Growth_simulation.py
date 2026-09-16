"""Growth simulation — historical start-date map of a contribution plan.

Sidebar page in the Streamlit visualizer:

  .venv/bin/streamlit run viz/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from viz.data import list_packages, load_package, strategy_map
from viz.growth_sim import (
    CAGR_HIST_LABELS,
    DEFAULT_LTCG,
    DEFAULT_ORDINARY,
    DEFAULT_REALIZED_FRACTION,
    DEFAULT_STATE,
    DEFAULT_STCG_FRACTION,
    apply_plan,
    blended_gain_rate,
    window_stats,
)

st.set_page_config(page_title="Growth simulation", layout="wide")


def _money(n: float) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    if abs(n) >= 1_000_000:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.0f}"


@st.cache_data(show_spinner=False)
def _windows(path: str, col: str, years: int, bench_col: str | None) -> dict:
    pkg = load_package(path)
    daily = pkg["daily"]
    bench = daily[bench_col] if bench_col and bench_col in daily.columns else None
    stats = window_stats(daily[col], years, bench)
    payload = stats.copy()
    if not payload.empty:
        payload["start"] = payload["start"].astype(str)
    return payload.to_dict(orient="list")


def _stats_df(payload: dict):
    import pandas as pd

    df = pd.DataFrame(payload)
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"])
    return df


packages = list_packages()
if not packages:
    st.error("No experiment packages found. Export one into `viz/packages/` first.")
    st.stop()

st.title("Growth simulation")
st.caption(
    "Every monthly start in the package mapped onto your lump + contribution plan "
    "at that window's realized CAGR. Not Monte Carlo. Re-export the package to refresh."
)

with st.sidebar:
    st.header("Series")
    pkg_labels = {}
    for p in packages:
        meta_preview = json.loads((p / "meta.json").read_text())
        pkg_labels[meta_preview.get("title", p.name)] = str(p)
    choice = st.selectbox("Package", list(pkg_labels.keys()))
    pkg = load_package(pkg_labels[choice])
    meta = pkg["meta"]
    smap = strategy_map(meta)
    cols = list(pkg["daily"].columns)
    primary = meta.get("primary_strategy") if meta.get("primary_strategy") in cols else cols[0]
    strategy = st.selectbox(
        "Strategy",
        cols,
        index=cols.index(primary),
        format_func=lambda i: smap.get(i, {}).get("name", i),
    )
    bench_opts = [c for c in cols if c != strategy]
    default_bench = next((c for c in ("qqq_bh", "spy_bh") if c in bench_opts), bench_opts[0] if bench_opts else None)
    bench_col = (
        st.selectbox(
            "Benchmark (win rate)",
            bench_opts,
            index=bench_opts.index(default_bench) if default_bench in bench_opts else 0,
            format_func=lambda i: smap.get(i, {}).get("name", i),
        )
        if bench_opts
        else None
    )
    st.caption(f"{meta.get('date_start')} → {meta.get('date_end')}")

    st.divider()
    st.header("Plan")
    start = st.slider("Starting capital ($)", 500, 50_000, 5_000, 500)
    monthly = st.slider("Monthly contribution ($)", 0, 1_500, 300, 25)
    years = st.slider("Horizon (years)", 3, 20, 10, 1)

    st.divider()
    st.header("Tax drag (taxable)")
    apply_tax = st.toggle("Show after-tax", value=True,
                          help="Taxable-account estimate. Off = tax-advantaged (Roth/401k).")
    realized = st.slider(
        "Gains realized per year", 0.0, 1.0, DEFAULT_REALIZED_FRACTION, 0.05,
        help="Share of each year's gain actually sold and taxed. A defer-heavy "
             "strategy that holds sleeves between CB/mode exits sits well below "
             "100%; 100% ≈ realizing everything annually (worst case).",
    )
    ordinary = st.slider("Federal short-term / ordinary rate", 0.0, 0.45, DEFAULT_ORDINARY, 0.01)
    ltcg = st.slider("Federal long-term cap-gains rate", 0.0, 0.25, DEFAULT_LTCG, 0.01)
    state_tax = st.slider("State rate", 0.0, 0.15, DEFAULT_STATE, 0.005)
    stcg_frac = st.slider(
        "Realized gains taxed as short-term", 0.0, 1.0, DEFAULT_STCG_FRACTION, 0.05,
        help="Held > 1 year → long-term (lower). Deferral pushes this down.",
    )
    blended = blended_gain_rate(ordinary, ltcg, state_tax, stcg_frac)
    st.caption(f"Blended rate on realized gains ≈ **{blended:.1%}**. "
               "Sensitivity model, not tax advice. See `viz/growth_sim.py`.")

stats = _stats_df(_windows(pkg_labels[choice], strategy, years, bench_col))
if stats.empty:
    st.warning("Not enough history for that horizon.")
    st.stop()

plan = apply_plan(
    stats, float(start), float(monthly), int(years),
    blended if apply_tax else 0.0, float(realized),
)
name = smap.get(strategy, {}).get("name", strategy)
bench_name = smap.get(bench_col, {}).get("name", bench_col) if bench_col else "—"
cash = plan["cash_in"]
med = plan["terminal"][50]
p10 = plan["terminal"][10]
p25 = plan["terminal"][25]
p75 = plan["terminal"][75]
p90 = plan["terminal"][90]
worst = plan["terminal"][0]
cagr = plan["cagr_pct"]
at_cagr = plan["at_cagr_pct"]
at_med = plan["at_terminal"][50]
at_drag = (1 - at_med / med) if med else float("nan")

info = (
    f"**Most likely (median):** {_money(med)} after {years} years at {cagr[50]:.1%} CAGR. "
    f"Central half of history: {_money(p25)} – {_money(p75)}. "
    f"{plan['n']} overlapping {years}-year month-starts in {name}."
)
if apply_tax:
    info += (
        f"  \n**After tax** (taxable, {blended:.0%} blended · {realized:.0%} realized/yr): "
        f"{_money(at_med)} median at {at_cagr[50]:.1%} CAGR — a **{at_drag:.0%}** haircut to ending wealth."
    )
st.info(info)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Median ending value", _money(med), f"{cagr[50]:.1%} CAGR")
m2.metric("Likely range (P25–P75)", f"{_money(p25)} – {_money(p75)}")
m3.metric("Bad case (P10)", _money(p10), f"{cagr[10]:.1%} CAGR")
m4.metric("Worst history", _money(worst), f"{cagr[0]:.1%} CAGR")

if apply_tax:
    t1, t2, t3 = st.columns(3)
    t1.metric("After-tax median", _money(at_med), f"{at_cagr[50]:.1%} CAGR")
    t2.metric("Tax haircut to wealth", f"−{at_drag:.0%}", f"−{_money(med - at_med)}",
              delta_color="inverse")
    t3.metric("After-tax bad case (P10)", _money(plan["at_terminal"][10]),
              f"{at_cagr[10]:.1%} CAGR")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=plan["path_p10"].index, y=plan["path_p10"].values, name="P10", line=dict(color="#d97706"))
    )
    fig.add_trace(
        go.Scatter(
            x=plan["path_median"].index,
            y=plan["path_median"].values,
            name="Median",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(x=plan["path_p90"].index, y=plan["path_p90"].values, name="P90", line=dict(color="#059669"))
    )
    if apply_tax:
        fig.add_trace(
            go.Scatter(
                x=plan["at_path_median"].index, y=plan["at_path_median"].values,
                name="Median (after tax)",
                line=dict(color="#2563eb", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        title=f"Wealth path — {name}, constant historical CAGRs",
        xaxis_title="Year",
        yaxis_title="Portfolio ($)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    cap = "Year 0 is the starting lump. Same cash flows; three realized-CAGR rates from the start-date grid."
    if apply_tax:
        cap += " Dashed line is the median after taxes on realized gains."
    st.caption(cap)

with right:
    hist = plan["hist_pct"]
    fig_h = go.Figure(go.Bar(x=list(CAGR_HIST_LABELS), y=list(hist.values), marker_color="#2563eb", name="% of starts"))
    fig_h.update_layout(
        title=f"Distribution of {years}-year CAGRs",
        xaxis_title="Annualized return bucket",
        yaxis_title="% of month-starts",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_h, use_container_width=True)
    beat_txt = (
        f"beat {bench_name} in {plan['beat_bench_pct']:.0f}% of windows"
        if not np.isnan(plan["beat_bench_pct"])
        else ""
    )
    st.caption(f"{plan['ge10_pct']:.0f}% of windows compounded at 10%+ · {beat_txt}".strip(" ·"))

labels = ["Worst", "P10", "P25", "Median", "P75", "P90", "Best"]
keys = [0, 10, 25, 50, 75, 90, 100]
fig_t = go.Figure()
fig_t.add_trace(
    go.Bar(
        x=labels,
        y=[plan["terminal"][k] / 1000 for k in keys],
        marker_color=("#94a3b8" if apply_tax else
                      ["#dc2626", "#d97706", "#64748b", "#2563eb", "#64748b", "#059669", "#059669"]),
        name="Pre-tax" if apply_tax else "Ending value ($k)",
    )
)
if apply_tax:
    fig_t.add_trace(
        go.Bar(
            x=labels,
            y=[plan["at_terminal"][k] / 1000 for k in keys],
            marker_color="#2563eb",
            name="After tax",
        )
    )
fig_t.update_layout(
    title="Ending-value percentiles ($ thousands)",
    xaxis_title="Historical percentile",
    yaxis_title="Ending value ($k)",
    barmode="group",
    height=340,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=40, r=20, t=60, b=40),
)
st.plotly_chart(fig_t, use_container_width=True)

st.subheader("Outcome table")
rows = [
    ("Worst history", cagr[0], worst, "Dot-com start or a short window that includes a crash"),
    ("P10 — bad", cagr[10], p10, "Only 1 in 10 historical starts did worse"),
    ("P25 — likely floor", cagr[25], p25, "Bottom of the central half"),
    ("Median — plan here", cagr[50], med, "Most useful single number"),
    ("P75 — likely ceiling", cagr[75], p75, "Top of the central half"),
    ("P90 — strong", cagr[90], p90, "Do not budget this"),
]
if apply_tax:
    rows.append((
        f"After-tax median ({blended:.0%} blended, {realized:.0%} realized/yr)",
        at_cagr[50],
        at_med,
        f"Taxable account. {at_drag:.0%} less ending wealth than the pre-tax median.",
    ))
    rows.append((
        "After-tax P10 — bad",
        at_cagr[10],
        plan["at_terminal"][10],
        "Taxable, unlucky start. Tax still owed in up years.",
    ))
st.dataframe(
    {
        "Scenario": [r[0] for r in rows],
        "CAGR": [f"{r[1]:.1%}" for r in rows],
        "Ending value": [_money(r[2]) for r in rows],
        "vs cash in": [_money(r[2] - cash) for r in rows],
        "How to read it": [r[3] for r in rows],
    },
    hide_index=True,
    use_container_width=True,
)

st.subheader("What else to know")
d0, d10, d50 = plan["maxdd_pct"][0], plan["maxdd_pct"][10], plan["maxdd_pct"][50]
k1, k2, k3 = st.columns(3)
k1.markdown(
    f"**Drawdowns inside the window**  \n"
    f"Median peak-to-trough: **{d50:.1%}**. P10: **{d10:.1%}**. "
    f"Worst history: **{d0:.1%}** (2000 top). A spending need can hit that hole."
)
k2.markdown(
    f"**Cash in vs outcome**  \n"
    f"You contribute **{_money(cash)}**. Median gain is **{_money(med - cash)}**. "
    f"The monthly contribution dominates a small lump over long horizons."
)
k3.markdown(
    f"**This is not independent luck**  \n"
    f"{plan['n']} overlapping windows reuse one {meta.get('date_start')}–{meta.get('date_end')} tape. "
    f"Constant CAGR ignores sequence. Not a forecast. Re-export the package to refresh."
)

if apply_tax:
    st.caption(
        f"**Tax method:** each year taxes {realized:.0%} of that year's gain at a "
        f"{blended:.1%} blended rate ({stcg_frac:.0%} short-term); the rest defers and "
        "compounds untaxed. No tax on losses and none modeled at final liquidation. "
        "It is a turnover sensitivity, not a 1099 — the realized-fraction slider is the "
        "main lever. V19d holds sleeves between CB/mode exits, so a value well under 100% "
        "is realistic; slide to 100% for the tax-everything-annually worst case."
    )

_tax_flag = (
    f" --tax --realized {realized:.2f} --stcg-frac {stcg_frac:.2f}" if apply_tax else ""
)
st.caption(
    f"Method: month-start CAGR from daily returns (`{strategy}`), then lump + monthly "
    f"grow-then-contribute. Engine: `viz/growth_sim.py`. "
    f"CLI: `.venv/bin/python -m viz.growth_sim --start {start} --monthly {monthly} "
    f"--years {years}{_tax_flag}`"
)
