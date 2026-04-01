"""Parameter sensitivity analysis.

Tests robustness of performance to perturbations of all key parameters.
A robust system should degrade gracefully, not cliff-edge.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product

from validation.backtest_harness import (
    BacktestConfig, BacktestResult, run_backtest, _load_data
)


@dataclass
class SensitivityResult:
    """Result from a single parameter perturbation."""
    param_name: str
    param_value: object
    result: BacktestResult


@dataclass
class SensitivitySuite:
    """Full sensitivity analysis results for one parameter."""
    param_name: str
    values_tested: list
    sharpes: list
    returns: list
    max_dds: list
    terminals: list
    base_sharpe: float
    max_sharpe_change: float  # worst % change from base
    is_robust: bool  # no single perturbation drops Sharpe >30%


def _make_config(base: BacktestConfig, param: str, value) -> BacktestConfig:
    """Create a modified config with one parameter changed."""
    d = {**base.__dict__}

    if param == "sma_periods":
        d["sma_periods"] = value
        d["faber_full_threshold"] = len(value)  # all SMAs for full weight
    elif param == "baseline_ivv":
        bl = dict(d["baseline"])
        delta = value - bl["IVV"]
        bl["IVV"] = value
        bl["cash"] = max(0.03, bl["cash"] - delta)
        # Renormalize
        total = sum(bl.values())
        if abs(total - 1.0) > 0.01:
            bl = {k: v / total for k, v in bl.items()}
        d["baseline"] = bl
    elif param == "baseline_qqq":
        bl = dict(d["baseline"])
        delta = value - bl["QQQ"]
        bl["QQQ"] = value
        bl["cash"] = max(0.03, bl["cash"] - delta)
        total = sum(bl.values())
        if abs(total - 1.0) > 0.01:
            bl = {k: v / total for k, v in bl.items()}
        d["baseline"] = bl
    elif param == "tier1_sub":
        ts = dict(d["tier_subs"])
        ts[1] = value
        d["tier_subs"] = ts
    elif param == "tier2_sub":
        ts = dict(d["tier_subs"])
        ts[2] = value
        d["tier_subs"] = ts
    elif param in d:
        d[param] = value
    else:
        raise ValueError(f"Unknown parameter: {param}")

    return BacktestConfig(**d)


# ── Parameter grid definitions ───────────────────────────────────────

SENSITIVITY_GRID = {
    # Harvey engine
    "harvey_similar_pctl": [0.10, 0.12, 0.15, 0.18, 0.20],
    "harvey_exclude_months": [24, 30, 36, 42, 48],
    "harvey_zscore_window": [80, 100, 120, 140, 160],

    # Faber SMA periods
    "sma_periods": [
        [6, 10, 12],       # production
        [3, 6, 10],        # shorter
        [6, 12, 18],       # wider spread
        [10, 12, 15],      # longer
        [6, 10],            # 2 SMAs
        [6, 10, 12, 18],   # 4 SMAs
    ],

    # Faber filter
    "faber_partial_mult": [0.50, 0.60, 0.70, 0.80, 0.90],

    # Baseline weights
    "baseline_ivv": [0.35, 0.40, 0.45, 0.50, 0.55],
    "baseline_qqq": [0.15, 0.20, 0.25, 0.30, 0.35],

    # Leverage tiers
    "tier1_sub": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "tier2_sub": [0.45, 0.55, 0.65, 0.75, 0.85],

    # Allocation constraints
    "max_single": [0.50, 0.55, 0.60, 0.65, 0.70],
    "max_equity": [0.75, 0.80, 0.85, 0.90, 0.95],

    # Weekly de-lever
    "weekly_delever_threshold": [0, 2, 3],  # 0=disabled, 2=2of3, 3=3of3
}


def run_sensitivity_single(param_name: str, values: list = None,
                            base_config: BacktestConfig = None,
                            data: dict = None,
                            verbose: bool = False) -> SensitivitySuite:
    """Run sensitivity analysis for a single parameter."""
    if base_config is None:
        base_config = BacktestConfig()
    if data is None:
        data = _load_data()
    if values is None:
        values = SENSITIVITY_GRID.get(param_name, [])

    # Run base
    base_result = run_backtest(base_config, data)
    base_sharpe = base_result.sharpe

    sharpes, returns, max_dds, terminals = [], [], [], []

    for val in values:
        try:
            cfg = _make_config(base_config, param_name, val)
            r = run_backtest(cfg, data)
            sharpes.append(r.sharpe)
            returns.append(r.ann_return)
            max_dds.append(r.max_dd)
            terminals.append(r.terminal_value)
            if verbose:
                print(f"  {param_name}={val}: Sharpe={r.sharpe:.3f}, "
                      f"Return={r.ann_return:.1%}, DD={r.max_dd:.1%}")
        except Exception as e:
            if verbose:
                print(f"  {param_name}={val}: FAILED ({e})")
            sharpes.append(float("nan"))
            returns.append(float("nan"))
            max_dds.append(float("nan"))
            terminals.append(float("nan"))

    # Compute max Sharpe change
    valid_sharpes = [s for s in sharpes if not np.isnan(s)]
    if valid_sharpes and base_sharpe > 0:
        max_change = max(abs(s / base_sharpe - 1) for s in valid_sharpes)
    else:
        max_change = float("nan")

    is_robust = (not np.isnan(max_change) and max_change < 0.30 and
                 all(s > 0 for s in valid_sharpes))

    return SensitivitySuite(
        param_name=param_name,
        values_tested=values,
        sharpes=sharpes,
        returns=returns,
        max_dds=max_dds,
        terminals=terminals,
        base_sharpe=base_sharpe,
        max_sharpe_change=max_change,
        is_robust=is_robust,
    )


def run_full_sensitivity(base_config: BacktestConfig = None,
                          data: dict = None,
                          params: list = None,
                          verbose: bool = True) -> dict[str, SensitivitySuite]:
    """Run sensitivity analysis across all parameters.

    Args:
        params: List of parameter names to test. None = all in SENSITIVITY_GRID.
    """
    if data is None:
        data = _load_data()
    if params is None:
        params = list(SENSITIVITY_GRID.keys())

    results = {}
    for i, param in enumerate(params):
        if verbose:
            print(f"\n[{i+1}/{len(params)}] Sensitivity: {param}")
        results[param] = run_sensitivity_single(param, data=data,
                                                 base_config=base_config,
                                                 verbose=verbose)

    return results


def interaction_test(param_pairs: list = None,
                      base_config: BacktestConfig = None,
                      data: dict = None,
                      verbose: bool = True) -> dict:
    """Test pairwise parameter interactions.

    Tests a few critical parameter pairs to ensure no adverse interactions.
    """
    if data is None:
        data = _load_data()
    if base_config is None:
        base_config = BacktestConfig()

    if param_pairs is None:
        param_pairs = [
            ("tier1_sub", "tier2_sub"),
            ("harvey_similar_pctl", "harvey_exclude_months"),
            ("baseline_ivv", "baseline_qqq"),
        ]

    base_result = run_backtest(base_config, data)
    results = {"base_sharpe": base_result.sharpe, "pairs": {}}

    for p1, p2 in param_pairs:
        v1s = SENSITIVITY_GRID.get(p1, [])
        v2s = SENSITIVITY_GRID.get(p2, [])

        # Test corners only (min/max of each)
        if len(v1s) < 2 or len(v2s) < 2:
            continue

        corners = [
            (v1s[0], v2s[0]),
            (v1s[0], v2s[-1]),
            (v1s[-1], v2s[0]),
            (v1s[-1], v2s[-1]),
        ]

        pair_results = []
        for v1, v2 in corners:
            try:
                cfg = _make_config(base_config, p1, v1)
                cfg = _make_config(cfg, p2, v2)
                r = run_backtest(cfg, data)
                pair_results.append({
                    p1: v1, p2: v2,
                    "sharpe": r.sharpe,
                    "ann_return": r.ann_return,
                    "max_dd": r.max_dd,
                })
                if verbose:
                    print(f"  {p1}={v1}, {p2}={v2}: Sharpe={r.sharpe:.3f}")
            except Exception as e:
                if verbose:
                    print(f"  {p1}={v1}, {p2}={v2}: FAILED ({e})")

        results["pairs"][(p1, p2)] = pair_results

    return results
