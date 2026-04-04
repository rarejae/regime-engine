"""Allocation schemes using regime-conditioned estimates.

Three approaches:
1. Inverse-vol weighted (Harvey-compatible baseline comparison)
2. Mean-variance optimal (uses full covariance information)
3. Risk parity on conditioned covariance (ignores return estimates)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def allocate_inverse_vol(
    expected_returns: pd.Series,
    conditioned_vols: pd.Series,
    eligible_assets: list[str],
    max_single: float = 0.40,
) -> dict[str, float]:
    """Inverse-vol weighted allocation — direct comparison to Harvey system.

    Score = max(ER, 0) / vol for each eligible asset.
    Allocate proportional to scores.

    Args:
        expected_returns: Conditioned expected returns per asset.
        conditioned_vols: Conditioned annualized volatilities.
        eligible_assets: Assets that pass the Faber filter.
        max_single: Maximum weight for any single asset.

    Returns:
        Weight dict for all assets including cash.
    """
    scores = {}
    for a in eligible_assets:
        if a == "cash":
            continue
        er = expected_returns.get(a, 0)
        vol = conditioned_vols.get(a, 0.15)
        if er > 0 and vol > 0.01:
            scores[a] = er / vol

    if not scores:
        return {a: (1.0 if a == "cash" else 0.0) for a in expected_returns.index}

    total = sum(scores.values())
    weights = {}
    for a in expected_returns.index:
        if a == "cash":
            weights[a] = 0.0
        elif a in scores:
            weights[a] = scores[a] / total
        else:
            weights[a] = 0.0

    # Cap single asset
    for a in weights:
        if a != "cash" and weights[a] > max_single:
            excess = weights[a] - max_single
            weights[a] = max_single
            weights["cash"] = weights.get("cash", 0) + excess

    # Remaining to cash
    allocated = sum(w for a, w in weights.items() if a != "cash")
    weights["cash"] = max(1.0 - allocated, 0)

    return weights


def allocate_mean_variance(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    eligible_assets: list[str],
    risk_aversion: float = 2.0,
    max_single: float = 0.40,
) -> dict[str, float]:
    """Mean-variance optimal allocation using conditioned inputs.

    Maximize: w' * mu - (gamma/2) * w' * Sigma * w
    Subject to: 0 <= w_i <= max_single, sum(w) <= 1

    Args:
        expected_returns: Conditioned expected returns.
        cov_matrix: Conditioned covariance matrix.
        eligible_assets: Assets that pass Faber filter.
        risk_aversion: Gamma parameter.
        max_single: Per-asset weight cap.

    Returns:
        Weight dict for all assets including cash.
    """
    risky = [a for a in eligible_assets if a != "cash"]
    if not risky:
        return {a: (1.0 if a == "cash" else 0.0) for a in expected_returns.index}

    mu = expected_returns.reindex(risky).fillna(0).values
    sigma = cov_matrix.reindex(index=risky, columns=risky).fillna(0).values
    n = len(risky)

    def neg_utility(w):
        return -(w @ mu - 0.5 * risk_aversion * w @ sigma @ w)

    def neg_utility_grad(w):
        return -(mu - risk_aversion * sigma @ w)

    # Constraints: sum of risky weights <= 0.97 (leave room for cash floor)
    constraints = [{"type": "ineq", "fun": lambda w: 0.97 - np.sum(w)}]
    bounds = [(0, max_single)] * n
    x0 = np.full(n, min(1.0 / n, max_single))

    result = minimize(
        neg_utility, x0, jac=neg_utility_grad,
        bounds=bounds, constraints=constraints, method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-10},
    )

    if not result.success:
        # Fallback to inverse-vol
        from experiments.kritzman_relevance.conditioned_estimates import conditioned_volatilities
        vols = conditioned_volatilities(cov_matrix)
        return allocate_inverse_vol(expected_returns, vols, eligible_assets, max_single)

    w_opt = np.maximum(result.x, 0)

    weights = {}
    for i, a in enumerate(risky):
        weights[a] = float(w_opt[i])
    for a in expected_returns.index:
        if a not in weights:
            weights[a] = 0.0
    weights["cash"] = max(1.0 - sum(w for a, w in weights.items() if a != "cash"), 0)

    return weights


def allocate_risk_parity(
    cov_matrix: pd.DataFrame,
    eligible_assets: list[str],
    max_single: float = 0.40,
) -> dict[str, float]:
    """Risk parity using conditioned covariance — equal risk contribution.

    Each risky asset contributes equally to total portfolio variance.
    Ignores expected returns entirely.

    Args:
        cov_matrix: Conditioned covariance matrix.
        eligible_assets: Assets passing Faber filter.
        max_single: Per-asset weight cap.

    Returns:
        Weight dict for all assets including cash.
    """
    risky = [a for a in eligible_assets if a != "cash"]
    if not risky:
        return {a: (1.0 if a == "cash" else 0.0) for a in cov_matrix.columns}

    sigma = cov_matrix.reindex(index=risky, columns=risky).fillna(0).values
    n = len(risky)

    def risk_budget_objective(w):
        """Minimize sum of squared differences in marginal risk contributions."""
        w = np.maximum(w, 1e-8)
        port_var = w @ sigma @ w
        if port_var < 1e-12:
            return 0.0
        mrc = sigma @ w  # marginal risk contribution
        rc = w * mrc  # risk contribution
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    bounds = [(0.01, max_single)] * n
    x0 = np.full(n, 1.0 / n)

    # Constraint: sum = target (leave room for cash)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 0.90}]

    result = minimize(
        risk_budget_objective, x0, bounds=bounds, constraints=constraints,
        method="SLSQP", options={"maxiter": 200, "ftol": 1e-12},
    )

    w_opt = np.maximum(result.x, 0) if result.success else np.full(n, 0.90 / n)

    # Normalize to sum to 0.90
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum() * 0.90

    weights = {}
    for i, a in enumerate(risky):
        weights[a] = float(min(w_opt[i], max_single))
    for a in cov_matrix.columns:
        if a not in weights:
            weights[a] = 0.0
    weights["cash"] = max(1.0 - sum(w for a, w in weights.items() if a != "cash"), 0)

    return weights


def apply_faber_filter(
    faber_scores: dict[str, int],
    baseline_weights: dict[str, float],
    partial_mult: float = 0.70,
) -> tuple[dict[str, float], float, list[str]]:
    """Faber trend filter — determines eligibility and pool size.

    Returns:
        (initial_weights, pool_size, eligible_assets)
    """
    weights = {}
    pool = 0.0
    eligible = []

    for a, base_w in baseline_weights.items():
        if a == "cash":
            weights[a] = base_w
            continue

        score = faber_scores.get(a, 0)
        if score >= 3:
            weights[a] = base_w
            eligible.append(a)
        elif score >= 2:
            weights[a] = base_w * partial_mult
            pool += base_w * (1.0 - partial_mult)
            eligible.append(a)
        else:
            weights[a] = 0.0
            pool += base_w

    return weights, pool, eligible


def distribute_pool(
    base_weights: dict[str, float],
    pool: float,
    asset_scores: dict[str, float],
    eligible: list[str],
    max_single: float = 0.40,
) -> dict[str, float]:
    """Distribute Faber pool to eligible assets proportional to scores.

    Args:
        base_weights: Weights after Faber filter.
        pool: Capital freed by Faber.
        asset_scores: Score per asset (from any allocation scheme).
        eligible: Assets eligible to receive pool capital.
        max_single: Per-asset cap.

    Returns:
        Final weight dict.
    """
    w = dict(base_weights)

    if pool <= 0.001:
        return w

    candidates = {a: asset_scores.get(a, 0) for a in eligible if a != "cash" and asset_scores.get(a, 0) > 0}

    if not candidates:
        w["cash"] = w.get("cash", 0) + pool
        return w

    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = max(min(pool, max_single - w.get(a, 0)), 0)
        w[a] = w.get(a, 0) + alloc
        w["cash"] = w.get("cash", 0) + (pool - alloc)
        return w

    total = sum(candidates.values())
    for a, sc in candidates.items():
        share = pool * (sc / total)
        w[a] = w.get(a, 0) + share

    return w
