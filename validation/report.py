"""Generate markdown robustness validation report."""

import numpy as np
from pathlib import Path
from datetime import datetime


def _fmt_pct(v, decimals=1):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v:+.{decimals}%}" if v < 0 else f"{v:.{decimals}%}"


def _fmt_f(v, decimals=2):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def _pf(passed):
    return "PASS" if passed else "**FAIL**"


def generate_report(sig_results: dict, wf_results: dict,
                     sens_results: dict, interaction_results: dict = None,
                     output_path: str = None) -> str:
    """Generate comprehensive markdown validation report.

    Args:
        sig_results: From statistical_significance.run_all_significance_tests
        wf_results: From walk_forward.run_all_walkforward
        sens_results: From sensitivity.run_full_sensitivity
        interaction_results: From sensitivity.interaction_test (optional)
        output_path: Write report to this path. None = don't write.

    Returns:
        Markdown string.
    """
    lines = []
    L = lines.append

    L("# TAA Robustness Validation Report")
    L(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L("")

    # ── Summary gate ─────────────────────────────────────────────
    L("## Pre-Deployment Gate Summary")
    L("")

    gates = []

    # Statistical significance gates
    sharpe_ci = sig_results["sharpe_ci"]
    gates.append(("Sharpe CI lower > 0", sig_results["sharpe_above_zero"]))
    gates.append(("Sharpe CI lower > 0.5", sig_results["sharpe_above_half"]))
    gates.append(("Sharpe > benchmark", sig_results["beats_benchmark"]))

    # Walk-forward gates
    gates.append(("All OOS windows Sharpe > 0", wf_results["all_oos_positive"]))
    gates.append(("Min OOS Sharpe > 0", wf_results["min_oos_sharpe_above_zero"]))
    gates.append(("Mean OOS Sharpe > 0.5", wf_results["mean_oos_sharpe_above_half"]))
    gates.append(("Split-sample stable (<50% decay)", wf_results["split_stable"]))

    # Sensitivity gates
    n_robust = sum(1 for s in sens_results.values() if s.is_robust)
    n_total = len(sens_results)
    gates.append((f"Sensitivity robust ({n_robust}/{n_total} params)", n_robust >= n_total * 0.7))

    all_pass = all(g[1] for g in gates)

    L("| Gate | Result |")
    L("|------|--------|")
    for name, passed in gates:
        L(f"| {name} | {_pf(passed)} |")
    L("")
    L(f"**Overall: {'PASS — cleared for deployment' if all_pass else 'FAIL — review issues below'}**")
    L("")

    # ── 1. Statistical Significance ──────────────────────────────
    L("## 1. Statistical Significance")
    L("")

    L("### Bootstrap Confidence Intervals (10k block-bootstrap, 21-day blocks)")
    L("")
    L("| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |")
    L("|--------|---------------|-------------|-------------|")

    sc = sig_results["sharpe_ci"]
    L(f"| Sharpe Ratio | {_fmt_f(sc.point_estimate)} | {_fmt_f(sc.ci_lower)} | {_fmt_f(sc.ci_upper)} |")

    tc = sig_results["terminal_ci"]
    L(f"| Terminal Wealth | ${_fmt_f(tc.point_estimate)} | ${_fmt_f(tc.ci_lower)} | ${_fmt_f(tc.ci_upper)} |")

    dc = sig_results["maxdd_ci"]
    L(f"| Max Drawdown | {_fmt_pct(dc.point_estimate)} | {_fmt_pct(dc.ci_lower)} | {_fmt_pct(dc.ci_upper)} |")
    L("")

    L("### Ledoit-Wolf Sharpe Ratio Test (vs benchmark)")
    lw = sig_results["lw_test"]
    L("")
    L(f"- Strategy Sharpe: {_fmt_f(lw.sharpe_a)}")
    L(f"- Benchmark Sharpe: {_fmt_f(lw.sharpe_b)}")
    L(f"- Difference: {_fmt_f(lw.sharpe_diff)}")
    L(f"- t-statistic: {_fmt_f(lw.t_stat)}")
    L(f"- p-value: {_fmt_f(lw.p_value, 4)}")
    L(f"- Significant at {lw.alpha:.0%}: {_pf(lw.significant)}")
    L("")

    # ── 2. Walk-Forward Validation ───────────────────────────────
    L("## 2. Walk-Forward Out-of-Sample Validation")
    L("")

    L("### Split-Sample Tests")
    L("")
    L("| Split | IS Sharpe | OOS Sharpe | Sharpe Decay |")
    L("|-------|----------|-----------|-------------|")
    for sp in wf_results["splits"]:
        decay_str = f"{sp.sharpe_decay:+.0%}" if not np.isnan(sp.sharpe_decay) else "N/A"
        L(f"| {sp.split_name} | {_fmt_f(sp.in_sample.sharpe)} | "
          f"{_fmt_f(sp.out_of_sample.sharpe)} | {decay_str} |")
    L("")

    L("### Expanding-Window Walk-Forward")
    wf = wf_results["walkforward"]
    L("")
    L("| Window | Test Period | OOS Sharpe | OOS Return |")
    L("|--------|------------|-----------|-----------|")
    for i, (train_end, test_start, test_end, result) in enumerate(wf.windows):
        L(f"| {i+1} | {test_start[:4]}-{test_end[:4]} | "
          f"{_fmt_f(result.sharpe)} | {_fmt_pct(result.ann_return)} |")
    L("")
    L(f"- Mean OOS Sharpe: {_fmt_f(wf.mean_oos_sharpe)}")
    L(f"- Std OOS Sharpe: {_fmt_f(wf.std_oos_sharpe)}")
    L(f"- Min OOS Sharpe: {_fmt_f(wf.min_oos_sharpe)}")
    L(f"- % Positive: {wf.pct_positive_sharpe:.0%}")
    L("")

    L("### Regime Stability")
    L("")
    regimes = wf_results["regimes"]
    L("| Regime | Sharpe | Ann Return | Max DD |")
    L("|--------|--------|-----------|--------|")
    for name, met in regimes.items():
        if name == "full":
            continue
        if met is not None:
            L(f"| {name} | {_fmt_f(met['sharpe'])} | {_fmt_pct(met['ann_return'])} | "
              f"{_fmt_pct(met['max_dd'])} |")
        else:
            L(f"| {name} | N/A | N/A | N/A |")
    L("")

    # ── 3. Parameter Sensitivity ─────────────────────────────────
    L("## 3. Parameter Sensitivity Analysis")
    L("")
    L("| Parameter | Values Tested | Base Sharpe | Min Sharpe | Max Sharpe | Max Change | Robust |")
    L("|-----------|-------------|------------|-----------|-----------|-----------|--------|")

    for param, suite in sorted(sens_results.items()):
        valid = [s for s in suite.sharpes if not np.isnan(s)]
        if valid:
            min_s = min(valid)
            max_s = max(valid)
        else:
            min_s = max_s = float("nan")

        n_vals = len(suite.values_tested)
        L(f"| {param} | {n_vals} | {_fmt_f(suite.base_sharpe)} | "
          f"{_fmt_f(min_s)} | {_fmt_f(max_s)} | "
          f"{suite.max_sharpe_change:+.0%} | {_pf(suite.is_robust)} |")
    L("")

    # Detailed per-parameter tables
    for param, suite in sorted(sens_results.items()):
        L(f"### {param}")
        L("")
        L("| Value | Sharpe | Ann Return | Max DD | Terminal |")
        L("|-------|--------|-----------|--------|---------|")
        for i, val in enumerate(suite.values_tested):
            sh = suite.sharpes[i] if i < len(suite.sharpes) else float("nan")
            ret = suite.returns[i] if i < len(suite.returns) else float("nan")
            dd = suite.max_dds[i] if i < len(suite.max_dds) else float("nan")
            tw = suite.terminals[i] if i < len(suite.terminals) else float("nan")
            val_str = str(val)
            L(f"| {val_str} | {_fmt_f(sh)} | {_fmt_pct(ret)} | "
              f"{_fmt_pct(dd)} | ${_fmt_f(tw)} |")
        L("")

    # ── 4. Interaction Tests ─────────────────────────────────────
    if interaction_results:
        L("## 4. Parameter Interaction Tests")
        L("")
        L(f"Base Sharpe: {_fmt_f(interaction_results['base_sharpe'])}")
        L("")

        for (p1, p2), corners in interaction_results.get("pairs", {}).items():
            L(f"### {p1} x {p2}")
            L("")
            L(f"| {p1} | {p2} | Sharpe | Ann Return | Max DD |")
            L("|------|------|--------|-----------|--------|")
            for c in corners:
                L(f"| {c[p1]} | {c[p2]} | {_fmt_f(c['sharpe'])} | "
                  f"{_fmt_pct(c['ann_return'])} | {_fmt_pct(c['max_dd'])} |")
            L("")

    # ── Footer ───────────────────────────────────────────────────
    L("---")
    L("*Report generated by validation/run_all.py*")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
