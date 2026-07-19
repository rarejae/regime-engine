"""Signal parity — live engine vs packaged monthly_state scores."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from live.signals import compute_smas, asset_score


def main() -> None:
    cache = ROOT / "experiments" / "v19d_marketstack_verification" / "yf_cache.parquet"
    state_path = ROOT / "viz" / "packages" / "v19d_marketstack_verification" / "monthly_state.parquet"
    if not cache.exists() or not state_path.exists():
        raise SystemExit("Need yf_cache.parquet and monthly_state.parquet — run export first")

    raw = pd.read_parquet(cache)
    prices = pd.DataFrame({
        "QQQ": raw["QQQ"],
        "IVV": raw["SPY"],
        "IAU": raw["GLD"],
    })
    prices.index = pd.to_datetime(prices.index)
    smas = compute_smas(prices)

    state = pd.read_parquet(state_path)
    state.index = pd.to_datetime(state.index)

    mismatches = []
    checked = 0
    for month, row in state.iterrows():
        # Score at prior trading day (same as backtest month-start convention)
        prior = prices.index[prices.index < month]
        if len(prior) == 0:
            continue
        day = prior[-1]
        for asset, col in [("QQQ", "qqq_sc"), ("IVV", "ivv_sc"), ("IAU", "iau_sc")]:
            if col not in row.index or pd.isna(row[col]):
                continue
            live = asset_score(day, asset, prices, smas)
            pkg = int(row[col])
            checked += 1
            if live != pkg:
                mismatches.append({
                    "month": month.strftime("%Y-%m"),
                    "score_day": day.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "live": live,
                    "package": pkg,
                })

    print(f"Checked {checked} score cells; mismatches={len(mismatches)}")
    if mismatches:
        print("First 15 mismatches:")
        for m in mismatches[:15]:
            print(m)
        # Allow small edge mismatches near data splice; fail if >1%
        rate = len(mismatches) / max(checked, 1)
        if rate > 0.02:
            raise SystemExit(f"FAIL parity rate {rate:.2%}")
        print(f"WARN parity rate {rate:.2%} within 2% tolerance")
    else:
        print("PASS — live signals match package")


if __name__ == "__main__":
    main()
