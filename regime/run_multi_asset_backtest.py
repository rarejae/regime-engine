"""Walk-forward multi-asset allocation backtest using regime similarity."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.multi_asset import (
    ASSETS, fetch_asset_returns,
    allocate_similarity_weighted, allocate_risk_parity_tilt,
    allocate_top_n, allocate_long_short,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TC = 0.0010  # 10bps round-trip transaction cost


def main(force: bool = False):
    config = RegimeConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  MULTI-ASSET REGIME ALLOCATION BACKTEST")
    print("=" * 80)

    # Load regime data
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)

    # Load asset returns
    asset_ret = fetch_asset_returns(force=force)
    print(f"\nAsset returns: {asset_ret.shape[0]} months × {asset_ret.shape[1]} assets")
    for col in asset_ret.columns:
        n = asset_ret[col].notna().sum()
        fv = asset_ret[col].first_valid_index()
        print(f"  {col:>15}: {n:>4} months (from {fv.date() if fv else 'N/A'})")

    # SPY forward returns for similarity lookups
    spy_fwd = asset_ret["us_equity"].shift(-1) if "us_equity" in asset_ret.columns else None

    # Backtest dates
    start = pd.Timestamp(config.backtest_start)
    end = pd.Timestamp(config.backtest_end)
    bt_dates = z_data.index[(z_data.index >= start) & (z_data.index <= end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]
    print(f"\nBacktest: {len(bt_dates)} months ({bt_dates.min().date()} to {bt_dates.max().date()})")

    # Compute rolling 12-month asset volatilities for risk parity
    asset_vol = asset_ret.rolling(12, min_periods=6).std()

    # Strategies
    strategies = {
        "similarity_weighted": {},
        "risk_parity_tilt": {},
        "top_3": {},
        "long_short": {},
        "benchmark_6040": {},
        "benchmark_ew": {},
        "benchmark_spy": {},
    }
    prev_weights = {s: None for s in strategies}

    for dt in bt_dates:
        # Available assets at this date
        avail = [a for a in ASSETS if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        if len(avail) < 3:
            continue

        actual_ret = {a: asset_ret.loc[dt, a] for a in avail if pd.notna(asset_ret.loc[dt, a])}

        # Regime similarity expected returns
        try:
            sim_result = compute_distances(z_data, dt, config)
        except ValueError:
            continue

        # Expected returns from similar months
        sim_er = {}
        dissim_er = {}
        for a in avail:
            if a not in asset_ret.columns:
                continue
            # Similar months' forward returns
            rets = [asset_ret.loc[d, a] for d in sim_result.similar_dates
                    if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
            sim_er[a] = np.mean(rets) if rets else 0.0

            # Dissimilar months' forward returns
            drets = [asset_ret.loc[d, a] for d in sim_result.dissimilar_dates
                     if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
            dissim_er[a] = np.mean(drets) if drets else 0.0

        vol_dict = {a: max(asset_vol.loc[dt, a], 0.001) if dt in asset_vol.index and a in asset_vol.columns and pd.notna(asset_vol.loc[dt, a]) else 0.10
                    for a in avail}

        # Compute allocations
        w1 = allocate_similarity_weighted(sim_er)
        w2 = allocate_risk_parity_tilt(sim_er, vol_dict)
        w3 = allocate_top_n(sim_er, n=3)
        w4 = allocate_long_short(sim_er, dissim_er)

        # Benchmarks
        w_6040 = {a: 0.0 for a in avail}
        if "us_equity" in avail:
            w_6040["us_equity"] = 0.60
        if "us_bonds" in avail:
            w_6040["us_bonds"] = 0.40
        w_ew = {a: 1.0 / len(avail) for a in avail}
        w_spy = {a: (1.0 if a == "us_equity" else 0.0) for a in avail}

        all_w = {
            "similarity_weighted": w1,
            "risk_parity_tilt": w2,
            "top_3": w3,
            "long_short": w4,
            "benchmark_6040": w_6040,
            "benchmark_ew": w_ew,
            "benchmark_spy": w_spy,
        }

        for sname, w in all_w.items():
            port_ret = sum(w.get(a, 0) * actual_ret.get(a, 0) for a in avail)

            # Transaction costs
            tc = 0.0
            if prev_weights[sname] is not None:
                turnover = sum(abs(w.get(a, 0) - prev_weights[sname].get(a, 0)) for a in avail)
                tc = turnover * TC / 2
            port_ret -= tc

            strategies[sname][dt] = port_ret
            prev_weights[sname] = w

    # Convert to Series
    results = {}
    for sname, rets_dict in strategies.items():
        if rets_dict:
            results[sname] = pd.Series(rets_dict).sort_index()

    # Report
    print(f"\n{'=' * 80}")
    print(f"  PERFORMANCE SUMMARY (1985-2024)")
    print(f"{'=' * 80}")

    print(f"\n  {'Strategy':>22} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'HitR':>5} {'CorrSPY':>8}")
    print(f"  {'-' * 68}")

    spy_rets = results.get("benchmark_spy")

    for sname in ["similarity_weighted", "risk_parity_tilt", "top_3", "long_short",
                   "benchmark_6040", "benchmark_ew", "benchmark_spy"]:
        s = results.get(sname)
        if s is None or len(s) < 12:
            print(f"  {sname:>22}: insufficient data")
            continue

        ann_ret = s.mean() * 12
        ann_vol = s.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + s).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        hit = (s > 0).mean()
        corr = s.corr(spy_rets) if spy_rets is not None and len(s) == len(spy_rets) else 0

        print(f"  {sname:>22} {ann_ret:>6.1%} {ann_vol:>6.1%} {sharpe:>7.2f} {max_dd:>7.1%} {hit:>5.0%} {corr:>7.2f}")

    # Decade performance
    print(f"\n{'=' * 80}")
    print(f"  PERFORMANCE BY DECADE")
    print(f"{'=' * 80}")

    decades = [("1985-1994", "1985", "1994"), ("1995-2004", "1995", "2004"),
               ("2005-2014", "2005", "2014"), ("2015-2024", "2015", "2024")]

    for dname, ds, de in decades:
        print(f"\n  {dname}:")
        print(f"  {'Strategy':>22} {'AnnRet':>7} {'Sharpe':>7}")
        print(f"  {'-' * 38}")
        for sname in ["similarity_weighted", "top_3", "benchmark_6040", "benchmark_spy"]:
            s = results.get(sname)
            if s is None:
                continue
            mask = (s.index >= pd.Timestamp(ds)) & (s.index < pd.Timestamp(f"{int(de)+1}"))
            dec = s[mask]
            if len(dec) < 12:
                continue
            ar = dec.mean() * 12
            av = dec.std() * np.sqrt(12)
            sh = ar / av if av > 0 else 0
            print(f"  {sname:>22} {ar:>6.1%} {sh:>7.2f}")

    # Crisis drawdowns
    print(f"\n{'=' * 80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'=' * 80}")

    crises = [("GFC", "2008-09", "2009-03"), ("COVID", "2020-02", "2020-03"),
              ("2022 Bear", "2022-01", "2022-10")]

    for cname, cs, ce in crises:
        print(f"\n  {cname} ({cs} to {ce}):")
        for sname in ["similarity_weighted", "top_3", "long_short",
                       "benchmark_6040", "benchmark_spy"]:
            s = results.get(sname)
            if s is None:
                continue
            mask = (s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))
            crisis = s[mask]
            if len(crisis) == 0:
                continue
            cum_ret = (1 + crisis).prod() - 1
            print(f"    {sname:>22}: {cum_ret:>+7.1%}")

    # Calendar year returns
    print(f"\n{'=' * 80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'=' * 80}")

    header = f"  {'Year':>6}"
    show = ["similarity_weighted", "top_3", "benchmark_6040", "benchmark_spy"]
    for sn in show:
        header += f" {sn[:12]:>12}"
    print(header)
    print(f"  {'-' * (6 + 13 * len(show))}")

    for year in range(1985, 2025):
        row = f"  {year:>6}"
        for sn in show:
            s = results.get(sn)
            if s is None:
                row += f" {'--':>12}"
                continue
            mask = s.index.year == year
            yr = s[mask]
            if len(yr) == 0:
                row += f" {'--':>12}"
            else:
                yr_ret = (1 + yr).prod() - 1
                row += f" {yr_ret:>+11.1%}"
        print(row)

    # Final values
    print(f"\n{'=' * 80}")
    print(f"  FINAL VALUES ($1 invested)")
    print(f"{'=' * 80}")
    for sname in ["similarity_weighted", "risk_parity_tilt", "top_3", "long_short",
                   "benchmark_6040", "benchmark_ew", "benchmark_spy"]:
        s = results.get(sname)
        if s is None or len(s) < 12:
            continue
        final = float((1 + s).cumprod().iloc[-1])
        print(f"  {sname:>22}: ${final:>8.2f}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
