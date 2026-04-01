"""Run full TAA robustness validation suite.

Usage:
    python -m validation.run_all [--quick] [--no-sensitivity] [--no-interactions]

Generates: reports/robustness_validation.md
"""

import sys, os, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from validation.backtest_harness import (
    BacktestConfig, run_backtest, run_benchmark, run_ivv_buyhold, _load_data
)
from validation.statistical_significance import run_all_significance_tests
from validation.walk_forward import run_all_walkforward
from validation.sensitivity import run_full_sensitivity, interaction_test, SENSITIVITY_GRID
from validation.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="TAA Robustness Validation")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced bootstrap samples (1000) and skip interactions")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Skip parameter sensitivity analysis")
    parser.add_argument("--no-interactions", action="store_true",
                        help="Skip parameter interaction tests")
    parser.add_argument("--params", nargs="+",
                        help="Only test these sensitivity parameters")
    parser.add_argument("--bootstrap-n", type=int, default=10000,
                        help="Number of bootstrap samples")
    args = parser.parse_args()

    if args.quick:
        args.bootstrap_n = 1000
        args.no_interactions = True

    print("=" * 80)
    print("  TAA ROBUSTNESS VALIDATION SUITE")
    print("=" * 80)

    t0 = time.time()

    # Load data once
    print("\n[1/5] Loading data...")
    data = _load_data()

    # Production config
    config = BacktestConfig()

    # ── Statistical Significance ─────────────────────────────────
    print(f"\n[2/5] Statistical significance (n={args.bootstrap_n})...")
    t1 = time.time()

    strat = run_backtest(config, data, verbose=True)
    bench = run_ivv_buyhold(data)

    sig_results = run_all_significance_tests(
        strat.daily_returns, bench.daily_returns,
        n_bootstrap=args.bootstrap_n,
    )

    sc = sig_results["sharpe_ci"]
    print(f"  Sharpe: {sc.point_estimate:.3f} [{sc.ci_lower:.3f}, {sc.ci_upper:.3f}]")
    print(f"  Sharpe > 0: {sig_results['sharpe_above_zero']}")
    print(f"  Sharpe > 0.5: {sig_results['sharpe_above_half']}")
    lw = sig_results["lw_test"]
    print(f"  Ledoit-Wolf: diff={lw.sharpe_diff:.3f}, p={lw.p_value:.4f}")
    print(f"  [{time.time()-t1:.0f}s]")

    # ── Walk-Forward ─────────────────────────────────────────────
    print(f"\n[3/5] Walk-forward validation...")
    t2 = time.time()

    wf_results = run_all_walkforward(config, data)

    wf = wf_results["walkforward"]
    print(f"  {len(wf.windows)} windows, mean OOS Sharpe: {wf.mean_oos_sharpe:.3f}")
    print(f"  Min OOS Sharpe: {wf.min_oos_sharpe:.3f}")
    print(f"  % positive: {wf.pct_positive_sharpe:.0%}")
    for sp in wf_results["splits"]:
        print(f"  {sp.split_name}: IS={sp.in_sample.sharpe:.3f}, OOS={sp.out_of_sample.sharpe:.3f}")
    print(f"  [{time.time()-t2:.0f}s]")

    # ── Parameter Sensitivity ────────────────────────────────────
    sens_results = {}
    if not args.no_sensitivity:
        print(f"\n[4/5] Parameter sensitivity...")
        t3 = time.time()

        params = args.params or list(SENSITIVITY_GRID.keys())
        sens_results = run_full_sensitivity(config, data, params=params, verbose=True)

        n_robust = sum(1 for s in sens_results.values() if s.is_robust)
        print(f"\n  Robust: {n_robust}/{len(sens_results)} parameters")
        print(f"  [{time.time()-t3:.0f}s]")
    else:
        print(f"\n[4/5] Parameter sensitivity... SKIPPED")

    # ── Interaction Tests ────────────────────────────────────────
    interaction_results = None
    if not args.no_interactions and not args.no_sensitivity:
        print(f"\n[5/5] Parameter interaction tests...")
        t4 = time.time()

        interaction_results = interaction_test(data=data, verbose=True)
        print(f"  [{time.time()-t4:.0f}s]")
    else:
        print(f"\n[5/5] Parameter interactions... SKIPPED")

    # ── Report ───────────────────────────────────────────────────
    output_path = "reports/robustness_validation.md"
    report = generate_report(sig_results, wf_results, sens_results,
                              interaction_results, output_path)

    total = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  COMPLETE ({total:.0f}s)")
    print(f"  Report: {output_path}")
    print(f"{'='*80}")

    # Print gate summary
    print(f"\n  PRE-DEPLOYMENT GATE:")
    gates = [
        ("Sharpe CI > 0", sig_results["sharpe_above_zero"]),
        ("Sharpe CI > 0.5", sig_results["sharpe_above_half"]),
        ("Beats benchmark", sig_results["beats_benchmark"]),
        ("All OOS Sharpe > 0", wf_results["all_oos_positive"]),
        ("Min OOS > 0", wf_results["min_oos_sharpe_above_zero"]),
        ("Mean OOS > 0.5", wf_results["mean_oos_sharpe_above_half"]),
        ("Split stable", wf_results["split_stable"]),
    ]
    if sens_results:
        n_robust = sum(1 for s in sens_results.values() if s.is_robust)
        n_total = len(sens_results)
        gates.append((f"Sensitivity ({n_robust}/{n_total})", n_robust >= n_total * 0.7))

    for name, passed in gates:
        status = "PASS" if passed else "FAIL"
        print(f"    {status:>4}  {name}")

    all_pass = all(g[1] for g in gates)
    print(f"\n  {'CLEARED FOR DEPLOYMENT' if all_pass else 'ISSUES FOUND — REVIEW REPORT'}")


if __name__ == "__main__":
    main()
