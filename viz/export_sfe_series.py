"""Export SFE principle-series packages for the Streamlit visualizer.

Writes:
  viz/packages/sfe_principle_series/   — SFE equal, 45/45/10, SFEv3+CB + benchmarks

Usage:
  .venv/bin/python viz/export_sfe_series.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.sfe_simple_faber_equal.backtest import (
    W_45_45_10,
    W_EQUAL,
    bh,
    load_prices,
    load_rfr,
    run_sfe,
    sixty_forty,
)
from experiments.sfev3_cb.backtest import run_sfev3
from viz.metrics import metrics_from_daily
from viz.schema import SCHEMA_VERSION

OUT = ROOT / "viz" / "packages" / "sfe_principle_series"

STRATEGIES = [
    {"id": "sfe_equal", "name": "SFE 1/3 equal", "kind": "strategy", "color": "#059669"},
    {"id": "sfe_454510", "name": "SFE 45/45/10", "kind": "strategy", "color": "#2563eb"},
    {"id": "sfev3", "name": "SFEv3 45/45/10+CB", "kind": "strategy", "color": "#dc2626"},
    {"id": "qqq_bh", "name": "QQQ B&H", "kind": "benchmark", "color": "#0ea5e9"},
    {"id": "spy_bh", "name": "SPY B&H", "kind": "benchmark", "color": "#64748b"},
    {"id": "sixty_forty", "name": "60/40", "kind": "benchmark", "color": "#f59e0b"},
]


def monthlyize(daily: pd.Series) -> pd.Series:
    return (
        daily.resample("MS")
        .apply(lambda x: (1 + x).prod() - 1 if len(x) else np.nan)
        .dropna()
    )


def state_from_log(log: pd.DataFrame) -> pd.DataFrame:
    """Map SFE monthly log → visualizer state columns."""
    s = log.copy()
    s["month"] = pd.to_datetime(s["month"])
    s = s.set_index("month").sort_index()
    s["eff_equity"] = s["w_qld"] * 2.0 + s["w_sso"] * 2.0
    # Faber binary → 0/3 score for the state chart
    s["qqq_sc"] = s["qqq_on"].astype(int) * 3
    s["ivv_sc"] = s["spy_on"].astype(int) * 3
    s["iau_sc"] = s["gld_on"].astype(int) * 3
    return s[["eff_equity", "qqq_sc", "ivv_sc", "iau_sc", "w_qld", "w_sso", "w_gld", "w_cash"]]


def main():
    print("Loading yfinance SFE universe + FRED cash...")
    px = load_prices()
    rfr = load_rfr(px.index)

    print("Running SFE equal / 45/45/10 / SFEv3...")
    sfe_eq, log_eq = run_sfe(px, rfr, weights=W_EQUAL)
    sfe_10, log_10 = run_sfe(px, rfr, weights=W_45_45_10)
    sfe_v3, log_v3, cbs = run_sfev3(px, rfr, weights=W_45_45_10)

    start = sfe_v3.index.min()
    idx = sfe_v3.index
    qqq = bh(px["QQQ"], start).reindex(idx).fillna(0.0)
    spy = bh(px["SPY"], start).reindex(idx).fillna(0.0)
    s6040 = sixty_forty(px["SPY"], rfr, start).reindex(idx).fillna(0.0)

    daily = pd.DataFrame(
        {
            "sfe_equal": sfe_eq.reindex(idx).fillna(0.0),
            "sfe_454510": sfe_10.reindex(idx).fillna(0.0),
            "sfev3": sfe_v3,
            "qqq_bh": qqq,
            "spy_bh": spy,
            "sixty_forty": s6040,
        }
    ).sort_index()

    monthly = pd.DataFrame({c: monthlyize(daily[c]) for c in daily.columns})
    # Primary state = SFEv3 (has CB); also write CB log
    state = state_from_log(log_v3)

    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT / "daily_returns.parquet")
    monthly.to_parquet(OUT / "monthly_returns.parquet")
    state.to_parquet(OUT / "monthly_state.parquet")
    if len(cbs):
        cb_df = cbs.copy()
        cb_df["date"] = pd.to_datetime(cb_df["date"])
        cb_df.to_csv(OUT / "cb_events.csv", index=False)

    print("\nSanity metrics:")
    for s in STRATEGIES:
        m = metrics_from_daily(daily[s["id"]])
        print(
            f"  {s['name']:<22} CAGR {m['CAGR']:.2%}  Sharpe {m['Sharpe']:.3f}  "
            f"MaxDD {m['MaxDD']:.1%}  Term ${m['Terminal_$1']:.2f}"
        )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "sfe_principle_series",
        "title": "SFE Principle Series",
        "description": (
            "A priori Faber principle tests: equal 1/3 sleeves, 45/45/10 gold cap, "
            "and SFEv3 (45/45/10 + daily 3/3 SMA CB→cash). Classic 10-month Faber, "
            "monthly rebalance, cash when off. yfinance + FRED DTB3."
        ),
        "generated": datetime.now(timezone.utc).isoformat(),
        "date_start": start.strftime("%Y-%m-%d"),
        "date_end": idx.max().strftime("%Y-%m-%d"),
        "primary_strategy": "sfev3",
        "strategies": STRATEGIES,
        "data_notes": (
            "yfinance adj closes (QQQ/SPY/GLD/QLD/SSO); FRED DTB3 cash. "
            "Pre-2006 QLD/SSO simulated as 2× underlying − rf − expense. "
            "CB: close < SMA 126/200/252 → cash next session; monthly re-entry."
        ),
        "source_experiment": "experiments/sfev3_cb",
        "related_notes": [
            "research/experiments/2026-07-22_sfe_simple_faber_equal.md",
            "research/experiments/2026-07-22_sfe_45_45_10_gold_cap.md",
            "research/experiments/2026-07-22_sfev3_cb.md",
        ],
        "cb_events_total": int(len(cbs)),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote package → {OUT}")
    print("Restart Streamlit (or wait for Cloud redeploy after push) to see it in the sidebar.")


if __name__ == "__main__":
    main()
