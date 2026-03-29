"""Main entry point for HMM v3.1 development."""

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from hmm.config import HMMConfig
from hmm.features import build_hmm_features, get_spy_prices
from hmm.model import fit_and_predict_rolling
from hmm.labeler import label_states, LabelingResult
from hmm.validation import walk_forward_validate
from hmm.analysis import compare_to_rules_based, generate_regime_timeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

CV_PATHS = [
    Path("data/macro/condition_vectors.pickle"),
    Path("condition_vectors.pickle"),
]


def main():
    config = HMMConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # === STEP 1 ===
    logger.info("=" * 60)
    logger.info(f"STEP 1: Building features ({len(config.hmm_features)} inputs)")
    logger.info("=" * 60)
    features = build_hmm_features(config)
    logger.info(f"Shape: {features.shape}, range: {features.index.min().date()} to {features.index.max().date()}")

    spy_prices = get_spy_prices(config)
    if spy_prices is None:
        logger.error("Cannot proceed without SPY prices"); sys.exit(1)

    # === STEP 2 ===
    logger.info("=" * 60)
    logger.info(f"STEP 2: Fitting {config.n_states}-state HMM (diag cov, rolling)")
    logger.info("=" * 60)
    predictions, last_model = fit_and_predict_rolling(features, config)
    logger.info(f"Predictions: {len(predictions)} dates")
    predictions.to_parquet(config.output_dir / "predictions.parquet", index=False)

    # === STEP 3 ===
    logger.info("=" * 60)
    logger.info("STEP 3: Labeling states (relative ranking, excess returns)")
    logger.info("=" * 60)
    result = label_states(predictions, features, spy_prices, config, last_model)
    profiles = result.profiles

    # Print profiles
    print(f"\n{'='*75}")
    print(f"  STATE PROFILES ({config.n_states} states, sorted by return)")
    print(f"  Unconditional mean 1m return: {result.unconditional_mean_1m:.2%}")
    print(f"  Confidence: median={result.confidence_median:.3f} p25={result.confidence_p25:.3f} p10={result.confidence_p10:.3f}")
    print(f"{'='*75}")

    for p in profiles:
        print(f"\n  State {p.state_id}: {p.label.upper()} [{p.direction}]")
        print(f"    Time:     {p.pct_of_time:.1%} of days")
        print(f"    Duration: avg={p.avg_duration_days:.0f}d  med={p.median_duration_days:.0f}d  "
              f"model-implied={p.model_implied_duration:.0f}d  self-trans={p.self_transition_prob:.3f}")
        if p.self_transition_prob > 0 and p.self_transition_prob < 0.90:
            print(f"    *** UNSTABLE: self-transition < 0.90 ***")
        print(f"    Return:   1m={p.avg_spy_return_1m:+.2%}  excess={p.excess_return_1m:+.2%}  "
              f"std={p.return_std_1m:.2%}  3m={p.avg_spy_return_3m:+.2%}")
        print(f"    Dir acc:  {p.direction_accuracy_1m:.1%} (on excess returns)")
        print(f"    Entry:    {p.entry_direction} ({p.avg_return_at_entry:+.3f})  "
              f"Late: {p.late_direction} ({p.avg_return_late:+.3f})")
        print(f"    Strategy: {p.recommended_strategy}")
        fstr = "  ".join(f"{k}={v:+.2f}" for k, v in sorted(p.feature_means.items()))
        print(f"    Features: {fstr}")
        for ex in p.example_periods[:3]:
            print(f"    Period:   {ex}")

    # Transition matrix
    if result.transition_matrix is not None:
        print(f"\n  TRANSITION MATRIX (last fit):")
        tm = result.transition_matrix
        header = "         " + "  ".join(f"  S{i}" for i in range(tm.shape[1]))
        print(f"  {header}")
        for i in range(tm.shape[0]):
            row = f"    S{i}  " + "  ".join(f"{tm[i,j]:.3f}" for j in range(tm.shape[1]))
            print(f"  {row}")

    # Validation checks
    print(f"\n  VALIDATION CHECKS:")
    bearish = [p for p in profiles if p.direction == "bearish"]
    print(f"    Bearish states: {len(bearish)} {'PASS' if bearish else 'FAIL'}")
    neg_excess = [p for p in profiles if p.excess_return_1m < 0]
    print(f"    States with negative excess return: {len(neg_excess)} {'PASS' if neg_excess else 'FAIL'}")
    small = [p for p in profiles if p.pct_of_time < 0.05]
    print(f"    States with <5% time: {len(small)} {'PASS' if not small else 'FAIL — ' + str([p.state_id for p in small])}")
    short = [p for p in profiles if p.avg_duration_days < 15]
    print(f"    States with avg dur <15d: {len(short)} {'PASS' if not short else 'WARN — ' + str([(p.state_id, f'{p.avg_duration_days:.0f}d') for p in short])}")

    # === STEP 4 ===
    logger.info("=" * 60)
    logger.info("STEP 4: Walk-forward validation")
    logger.info("=" * 60)
    val = walk_forward_validate(features, spy_prices, config)

    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD VALIDATION ({val.n_folds} folds)")
    print(f"{'='*75}")
    print(f"  Avg BIC: {val.avg_bic:,.0f}  AIC: {val.avg_aic:,.0f}")
    print(f"  Avg test LL: {val.avg_test_log_likelihood:.2f} (std {val.test_ll_std:.2f})")
    print(f"  Avg duration: {val.avg_state_duration:.1f}d (std {val.duration_std:.1f})")
    print(f"  Avg states used: {val.avg_states_used:.1f}/{config.n_states}")
    print(f"  Avg return separation: {val.avg_return_separation:.3f}")

    # === STEP 5 ===
    logger.info("=" * 60)
    logger.info("STEP 5: Comparing to rules-based classifier")
    logger.info("=" * 60)
    cv_path = next((p for p in CV_PATHS if p.exists()), None)

    if cv_path:
        with open(cv_path, "rb") as f:
            rules_cvs = pickle.load(f)
        comp = compare_to_rules_based(predictions, rules_cvs, spy_prices, profiles)

        print(f"\n{'='*75}")
        print(f"  HMM vs RULES-BASED COMPARISON")
        print(f"{'='*75}")
        print(f"  Comparable dates: {comp['total_comparable']}")
        print(f"  Agreement:        {comp['agreement_rate']:.1%}")
        print(f"  HMM accuracy:     {comp['hmm_accuracy_1m']:.1%}")
        print(f"  Rules accuracy:   {comp['rules_accuracy_1m']:.1%}")
        print(f"  HMM advantage:    {comp['hmm_advantage']:+.1%}")
        d = comp["disagreements"]
        if len(d) > 0:
            print(f"  Disagreements:    {len(d)} — HMM right: {d['hmm_right'].sum()}, Rules right: {d['rules_right'].sum()}")
    else:
        logger.warning("condition_vectors.pickle not found — skipping")

    # === STEP 6 ===
    tl = generate_regime_timeline(predictions, profiles, spy_prices)
    tl.to_csv(config.output_dir / "regime_timeline.csv", index=False)

    print(f"\n{'='*75}")
    print(f"  COMPLETE — output in {config.output_dir}/")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
