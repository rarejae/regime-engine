"""Main entry point for HMM development — fit, label, validate, compare, report."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hmm.config import HMMConfig
from hmm.features import build_hmm_features, get_spy_prices
from hmm.model import fit_and_predict_rolling
from hmm.labeler import label_states
from hmm.validation import walk_forward_validate
from hmm.analysis import compare_to_rules_based, generate_regime_timeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = HMMConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Build features ===
    logger.info("=" * 60)
    logger.info("STEP 1: Building HMM features")
    logger.info("=" * 60)

    features = build_hmm_features(config)
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Date range: {features.index.min().date()} to {features.index.max().date()}")

    raw = pd.read_parquet(config.raw_indicators_path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    spy_prices = get_spy_prices(raw, config)

    # === STEP 2: Fit HMM (rolling window) ===
    logger.info("=" * 60)
    logger.info("STEP 2: Fitting HMM with rolling window")
    logger.info("=" * 60)

    predictions = fit_and_predict_rolling(features, config)
    logger.info(f"Predictions: {len(predictions)} dates")
    predictions.to_parquet(config.output_dir / "predictions.parquet", index=False)

    # === STEP 3: Label states ===
    logger.info("=" * 60)
    logger.info("STEP 3: Labeling states")
    logger.info("=" * 60)

    profiles = label_states(predictions, features, spy_prices, config)

    print(f"\n{'='*70}")
    print(f"  STATE PROFILES ({config.n_states} states)")
    print(f"{'='*70}")
    for p in profiles:
        print(f"\n  State {p.state_id}: {p.label.upper()}")
        print(f"    Time:      {p.pct_of_time:.1%} of days | Avg run: {p.avg_duration_days:.0f} days")
        print(f"    Direction: {p.direction} | Dir accuracy: {p.direction_accuracy_1m:.1%}")
        print(f"    SPY fwd:   1m {p.avg_spy_return_1m:+.2%} (std {p.return_std_1m:.2%}) | 3m {p.avg_spy_return_3m:+.2%}")
        print(f"    Strategy:  {p.recommended_strategy}")
        for feat, val in sorted(p.feature_means.items()):
            print(f"    {feat}: {val:+.3f}")
        for ex in p.example_periods[:3]:
            print(f"    Period: {ex}")

    # === STEP 4: Walk-forward validation ===
    logger.info("=" * 60)
    logger.info("STEP 4: Walk-forward validation")
    logger.info("=" * 60)

    val_report = walk_forward_validate(features, spy_prices, config)

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION ({val_report.n_folds} folds)")
    print(f"{'='*70}")
    print(f"  Avg BIC:              {val_report.avg_bic:,.0f}")
    print(f"  Avg AIC:              {val_report.avg_aic:,.0f}")
    print(f"  Avg test LL:          {val_report.avg_test_log_likelihood:.3f}")
    print(f"  Avg state duration:   {val_report.avg_state_duration:.1f} days")
    print(f"  Avg states used/fold: {val_report.avg_states_used:.1f}")
    print(f"  Test LL std:          {val_report.direction_accuracy_std:.3f}")

    # === STEP 5: Compare to rules-based ===
    logger.info("=" * 60)
    logger.info("STEP 5: Comparing to rules-based classifier")
    logger.info("=" * 60)

    try:
        from backtest.macro_backfill import load_cached_condition_vectors
        rules_cvs = load_cached_condition_vectors()

        if rules_cvs:
            comparison = compare_to_rules_based(predictions, rules_cvs, spy_prices, profiles)

            print(f"\n{'='*70}")
            print(f"  HMM vs RULES-BASED COMPARISON")
            print(f"{'='*70}")
            print(f"  Comparable dates:   {comparison['total_comparable']}")
            print(f"  Agreement rate:     {comparison['agreement_rate']:.1%}")
            print(f"  HMM accuracy (1m):  {comparison['hmm_accuracy_1m']:.1%}")
            print(f"  Rules accuracy (1m):{comparison['rules_accuracy_1m']:.1%}")
            print(f"  HMM advantage:      {comparison['hmm_advantage']:+.1%}")

            disag = comparison["disagreements"]
            if len(disag) > 0:
                hmm_wins = disag["hmm_right"].sum()
                rules_wins = disag["rules_right"].sum()
                print(f"  Disagreements:      {len(disag)}")
                print(f"    HMM was right:    {hmm_wins}")
                print(f"    Rules was right:  {rules_wins}")
    except Exception as e:
        logger.warning(f"Could not compare to rules-based: {e}")

    # === STEP 6: Regime timeline ===
    logger.info("=" * 60)
    logger.info("STEP 6: Regime timeline")
    logger.info("=" * 60)

    timeline = generate_regime_timeline(predictions, profiles, spy_prices)
    timeline.to_csv(config.output_dir / "regime_timeline.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  MONTHLY REGIME TIMELINE")
    print(f"{'='*70}")
    print(f"{'Date':>10} {'State':>6} {'Label':>18} {'Conf':>6} {'Dir':>8} {'SPY_1m':>8}")
    print("-" * 60)
    for _, row in timeline.iterrows():
        r1 = f"{row['spy_1m_fwd']:+.1%}" if pd.notna(row["spy_1m_fwd"]) else "N/A"
        print(f"{row['date'].strftime('%Y-%m'):>10} {row['state']:>6} {row['label']:>18} "
              f"{row['confidence']:>6.2f} {row['direction']:>8} {r1:>8}")

    print(f"\n{'='*70}")
    print(f"  HMM DEVELOPMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Output saved to {config.output_dir}/")
    print()


if __name__ == "__main__":
    main()
