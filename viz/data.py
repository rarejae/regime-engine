"""Package discovery/load for the Streamlit visualizer and growth sim."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from viz.schema import REQUIRED_FILES

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"


def list_packages() -> list[Path]:
    if not PACKAGES.exists():
        return []
    out = []
    for p in sorted(PACKAGES.iterdir()):
        if p.is_dir() and all((p / f).exists() for f in REQUIRED_FILES):
            out.append(p)
    return out


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
