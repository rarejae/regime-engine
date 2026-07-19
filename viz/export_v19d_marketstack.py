"""Export a viz package from the V19d Marketstack verification data.

Usage:
  .venv/bin/python viz/export_v19d_marketstack.py

Writes viz/packages/v19d_marketstack_verification/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.v19d_marketstack_verification.backtest import load_spliced_data
import experiments.v19d_final.backtest as v19d_mod
from experiments.v19d_final.backtest import run_v19d_full
from viz.metrics import metrics_from_daily, dca_terminal
from viz.schema import SCHEMA_VERSION

OUT = ROOT / "viz" / "packages" / "v19d_marketstack_verification"

STRATEGIES = [
    {"id": "v19d", "name": "V19d", "kind": "strategy", "color": "#2563eb"},
    {"id": "ivv_bh", "name": "IVV B&H", "kind": "benchmark", "color": "#64748b"},
    {"id": "qqq_bh", "name": "QQQ B&H", "kind": "benchmark", "color": "#0ea5e9"},
    {"id": "blend_5050", "name": "50/50 IVV/QQQ", "kind": "benchmark", "color": "#8b5cf6"},
    {"id": "sixty_forty", "name": "60/40 (IVV/TLT)", "kind": "benchmark", "color": "#f59e0b"},
]


def monthlyize(daily: pd.Series) -> pd.Series:
    return daily.resample("MS").apply(lambda x: (1 + x).prod() - 1 if len(x) else np.nan).dropna()


def main():
    print("Loading spliced Marketstack/yfinance data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start = load_spliced_data()
    end_date = daily_ret.dropna(how="all").index.max()
    v19d_mod.END_DATE = end_date.strftime("%Y-%m-%d")
    start = "2000-01-01"

    print(f"Running V19d {start} → {end_date.date()}...")
    v19d_s, cb, ml, rebal, _ = run_v19d_full(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start
    )

    ivv = daily_ret["IVV"].loc[start:end_date].dropna()
    qqq = daily_ret["QQQ"].loc[start:end_date].dropna()
    # Align benchmarks to V19d calendar so metrics share the same window
    idx = v19d_s.index
    ivv = ivv.reindex(idx).fillna(0.0)
    qqq = qqq.reindex(idx).fillna(0.0)
    blend = 0.5 * ivv + 0.5 * qqq

    # 60/40: IVV + TLT (VGLT proxy). Before TLT inception (2002-07), hold IVV weight
    # in cash (rfr) for the bond sleeve so the series is defined from 2000.
    vglt = daily_ret["VGLT"].reindex(idx)
    rfr = rfr_daily.reindex(idx).fillna(0.0)
    bond = vglt.where(vglt.notna(), rfr)
    ivv_raw = daily_ret["IVV"].reindex(idx).fillna(0.0)
    sixty_forty = 0.60 * ivv_raw + 0.40 * bond.fillna(0.0)

    daily = pd.DataFrame({
        "v19d": v19d_s,
        "ivv_bh": ivv,
        "qqq_bh": qqq,
        "blend_5050": blend,
        "sixty_forty": sixty_forty,
    }).sort_index()

    monthly = pd.DataFrame({c: monthlyize(daily[c]) for c in daily.columns})

    state = pd.DataFrame(ml)
    if len(state):
        state["month"] = pd.to_datetime(state["month"])
        state = state.set_index("month").sort_index()

    cb_df = pd.DataFrame(cb)
    if len(cb_df):
        cb_df["date"] = pd.to_datetime(cb_df["date"])

    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT / "daily_returns.parquet")
    monthly.to_parquet(OUT / "monthly_returns.parquet")
    if len(state):
        state.to_parquet(OUT / "monthly_state.parquet")
    if len(cb_df):
        cb_df.to_csv(OUT / "cb_events.csv", index=False)

    # Sanity metrics printed for verification against experiment note
    print("\nSanity metrics (daily, full window):")
    for sid, name in [(s["id"], s["name"]) for s in STRATEGIES]:
        m = metrics_from_daily(daily[sid])
        dca = dca_terminal(monthly[sid], 21000, 700, convention="vault")
        print(
            f"  {name:<20} CAGR {m['CAGR']:.2%}  Sharpe {m['Sharpe']:.3f}  "
            f"MaxDD {m['MaxDD']:.1%}  Term ${m['Terminal_$1']:.2f}  "
            f"DCA(vault) ${dca/1e6:.2f}M"
        )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "v19d_marketstack_verification",
        "title": "V19d Marketstack Verification",
        "description": (
            "V19d (two-pod + gold, CB→cash) verified on Marketstack EOD data "
            "spliced with yfinance before 2016-08. Includes standard baselines."
        ),
        "generated": datetime.now(timezone.utc).isoformat(),
        "date_start": start,
        "date_end": end_date.strftime("%Y-%m-%d"),
        "primary_strategy": "v19d",
        "strategies": STRATEGIES,
        "data_notes": (
            "Marketstack 2016-08→present; yfinance before. Proxies: IVV→SPY, "
            "IAU→GLD, VGLT→TLT. 60/40 uses TLT; pre-TLT bond sleeve earns T-bills."
        ),
        "source_experiment": "experiments/v19d_marketstack_verification",
        "related_note": "research/experiments/2026-07-18_marketstack_verification.md",
        "rebalances": int(rebal),
        "cb_events_total": int(len(cb)),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote package → {OUT}")


if __name__ == "__main__":
    main()
