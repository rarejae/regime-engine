"""Walk-forward backtest of Harvey-Mulliner regime similarity signal."""

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
from regime.similarity import compute_distances, compute_all_similarities
from regime.analysis import build_signals, backtest_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(force: bool = False):
    config = RegimeConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Fetch data ===
    print("=" * 80)
    print("  HARVEY-MULLINER REGIME SIMILARITY ENGINE")
    print("=" * 80)

    raw = fetch_monthly_history(config, force=force)
    print(f"\nRaw data: {raw.shape[0]} months × {raw.shape[1]} variables")
    print(f"Range: {raw.index.min().date()} to {raw.index.max().date()}")
    for col in raw.columns:
        n = raw[col].notna().sum()
        fv = raw[col].first_valid_index()
        print(f"  {col:>20}: {n:>4} months (from {fv.date() if fv else 'N/A'})")

    # === STEP 2: Transform ===
    print(f"\n{'='*80}")
    print("  TRANSFORMING: 12-month change → 10yr rolling z-score → winsorize ±3")
    print(f"{'='*80}")

    transformed = transform_variables(raw, config)
    z_data = get_valid_zscored(transformed, config)
    print(f"Z-scored data: {z_data.shape[0]} months × {z_data.shape[1]} variables")
    print(f"Valid from: {z_data.index.min().date()} to {z_data.index.max().date()}")

    # Validation: autocorrelation and cross-correlations
    print(f"\nAutocorrelation (1-month) of z-scored variables:")
    for col in z_data.columns:
        ac = z_data[col].autocorr(lag=1)
        flag = "" if 0.80 <= ac <= 0.98 else " ← outside expected range"
        print(f"  {col:>20}: {ac:.3f}{flag}")

    print(f"\nCross-correlation matrix (max pairwise |r|):")
    corr = z_data.corr()
    max_r = 0; max_pair = ""
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            r = abs(corr.iloc[i, j])
            if r > max_r:
                max_r = r
                max_pair = f"{corr.index[i]} × {corr.columns[j]}"
    print(f"  Max |r| = {max_r:.3f} ({max_pair})")

    # === STEP 3: Compute similarities ===
    print(f"\n{'='*80}")
    print(f"  COMPUTING SIMILARITIES ({config.backtest_start} to {config.backtest_end})")
    print(f"{'='*80}")

    results = compute_all_similarities(z_data, config)
    print(f"Computed: {len(results)} monthly assessments")

    # === STEP 4: Known event validation ===
    print(f"\n{'='*80}")
    print("  VALIDATION: KNOWN MACRO EVENTS")
    print(f"{'='*80}")

    events = [
        ("2009-01-01", "GFC trough", "Should match 1980s recessions"),
        ("2022-08-01", "Peak 2022 inflation", "Should match late 1970s"),
        ("2020-03-01", "COVID crash", "Should show few clear analogs"),
        ("2011-08-01", "US downgrade", "Should match stress episodes"),
        ("2017-06-01", "Low vol goldilocks", "Should match 1990s calm"),
    ]

    for ds, label, expectation in events:
        dt = pd.Timestamp(ds)
        if dt not in z_data.index:
            # Find nearest
            candidates = z_data.index[z_data.index <= dt + pd.DateOffset(months=1)]
            if len(candidates) == 0:
                print(f"\n  {label} ({ds}): NO DATA")
                continue
            dt = candidates[-1]

        try:
            r = compute_distances(z_data, dt, config)
        except ValueError as e:
            print(f"\n  {label} ({ds}): {e}")
            continue

        top5 = r.distances.nsmallest(5)
        print(f"\n  {label} ({dt.date()}):")
        print(f"    Expectation: {expectation}")
        print(f"    Min distance: {r.min_distance:.2f}, Median: {r.median_distance:.2f}")
        print(f"    Target z: {', '.join(f'{v:+.2f}' for v in r.target_z)}")
        print(f"    Top 5 similar months:")
        for d, dist in top5.items():
            z_vals = z_data.loc[d].values
            print(f"      {d.date()} (d={dist:.2f}): z=[{', '.join(f'{v:+.2f}' for v in z_vals)}]")

    # === STEP 5: Backtest ===
    print(f"\n{'='*80}")
    print("  BACKTEST: SIMILARITY SIGNAL ON SPY")
    print(f"{'='*80}")

    signals = build_signals(results, raw, config)
    bt = backtest_signal(signals)

    print(f"\n  Period: {bt['start']} to {bt['end']} ({bt['n_months']} months)")
    print(f"\n  {'Metric':>25} {'Strategy':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Sharpe ratio':>25} {bt['strategy_sharpe']:>12.2f} {bt['buyhold_sharpe']:>12.2f}")
    print(f"  {'Ann. return':>25} {bt['strategy_ann_ret']:>11.1f}% {bt['buyhold_ann_ret']:>11.1f}%")
    print(f"  {'Hit rate':>25} {bt['hit_rate']:>11.0%}")
    print(f"  {'Max drawdown':>25} {bt['max_drawdown']:>11.1%}")
    print(f"  {'Corr to buy&hold':>25} {bt['corr_to_buyhold']:>12.2f}")
    print(f"  {'% months long':>25} {bt['pct_long']:>11.0%}")
    print(f"  {'Spread Sharpe':>25} {bt['spread_sharpe']:>12.2f}")
    print(f"  {'Final value ($1)':>25} ${bt['final_strategy_value']:>11.2f} ${bt['final_bh_value']:>11.2f}")

    # Anti-regime signal
    signals["anti_signal"] = np.where(signals["dissimilar_avg_fwd"] > 0, -1, 1)
    signals["anti_ret"] = signals["anti_signal"] * signals["spy_fwd_1m"]
    anti_sharpe = signals["anti_ret"].mean() / signals["anti_ret"].std() * np.sqrt(12) if signals["anti_ret"].std() > 0 else 0
    print(f"\n  Anti-regime signal Sharpe: {anti_sharpe:.2f}")

    # Spread signal: go long similar, short dissimilar
    signals["spread_ret"] = (signals["similar_avg_fwd"] - signals["dissimilar_avg_fwd"])
    print(f"  Similar-minus-dissimilar spread: avg={signals['spread'].mean():.4f}/mo, "
          f"std={signals['spread'].std():.4f}")

    # Save
    signals.to_csv(config.output_dir / "similarity_signals.csv", index=False)

    # === SUMMARY ===
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  Strategy Sharpe: {bt['strategy_sharpe']:.2f} (buy-and-hold: {bt['buyhold_sharpe']:.2f})")
    print(f"  Hit rate: {bt['hit_rate']:.0%}")
    ok = bt['strategy_sharpe'] > 0
    print(f"  Positive Sharpe: {'PASS' if ok else 'FAIL'}")
    print(f"  Harvey paper reported 0.82 on 6-factor portfolio; single-asset SPY expected lower")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch all data")
    args = parser.parse_args()
    main(force=args.force)
