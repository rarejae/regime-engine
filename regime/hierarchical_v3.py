"""Hierarchical v3: multi-timeframe Faber + inverse-vol Harvey + HMM + Kritzman."""

import numpy as np
import pandas as pd

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
EQUITY = ["IVV", "QQQ"]
ASSETS = list(BASELINE.keys())
SMA_PERIODS = [6, 10, 12]


def compute_multi_sma_signals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend strength (0-3) for each asset from 3 SMAs.

    Returns DataFrame of integers (0, 1, 2, 3) indexed by month-start.
    """
    monthly = prices_df.resample("MS").last()
    result = pd.DataFrame(0, index=monthly.index, columns=monthly.columns)

    for period in SMA_PERIODS:
        sma = monthly.rolling(period, min_periods=period).mean()
        above = (monthly > sma).astype(int)
        result += above

    return result


def step1_multi_faber(trend_strengths: dict) -> tuple[dict, float, dict]:
    """Step 1: Multi-timeframe Faber gatekeeper.

    3/3 above: full baseline weight.
    2/3 above: 70% of baseline weight, 30% to pool.
    0-1/3 above: ineligible, full weight to pool.

    Returns (weights, pool_size, strength_detail).
    """
    w = {}
    pool = 0.0
    detail = {}

    for a in ASSETS:
        if a == "cash":
            w[a] = BASELINE[a]
            continue

        strength = trend_strengths.get(a, 0)
        detail[a] = strength

        if strength >= 3:
            w[a] = BASELINE[a]
        elif strength == 2:
            w[a] = BASELINE[a] * 0.70
            pool += BASELINE[a] * 0.30
        else:
            w[a] = 0.0
            pool += BASELINE[a]

    return w, pool, detail


def step2_harvey_invvol(
    weights: dict,
    pool: float,
    harvey_er: dict,
    realized_vols: dict,
) -> tuple[dict, dict]:
    """Step 2: Harvey capital allocation weighted by inverse volatility.

    score_i = expected_return_i / realized_vol_i (for positive ER only).
    Distribute pool proportional to scores.
    """
    detail = {"pool": pool, "scores": {}, "directed_to": {}}

    if pool <= 0.001:
        return weights, detail

    # Eligible assets with positive expected return
    candidates = {}
    for a in ASSETS:
        if a == "cash":
            continue
        if weights.get(a, 0) <= 0:
            continue  # ineligible
        er = harvey_er.get(a, 0)
        vol = realized_vols.get(a, 0.15)
        if er > 0 and vol > 0.01:
            score = er / vol
            candidates[a] = score
            detail["scores"][a] = {"er": er, "vol": vol, "score": score}

    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool
        detail["directed_to"]["cash"] = pool
        return weights, detail

    total_score = sum(candidates.values())
    for a, score in candidates.items():
        share = pool * (score / total_score)
        weights[a] = weights.get(a, 0) + share
        detail["directed_to"][a] = share

    return weights, detail


def step3_hmm(weights: dict, bull_prob: float) -> tuple[dict, str]:
    """Step 3: HMM conviction (unchanged from v1)."""
    if bull_prob > 0.7:
        action = "boost_15pct"
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                boost = weights[a] * 0.15
                weights[a] += boost
                weights["cash"] = max(weights.get("cash", 0) - boost, 0)
    elif bull_prob < 0.3:
        action = "reduce_15pct"
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                red = weights[a] * 0.15
                weights[a] -= red
                weights["cash"] = weights.get("cash", 0) + red
    else:
        action = "no_change"
    return weights, action


def step4_kritzman(weights: dict, turb_pctl: float, ar_z: float) -> tuple[dict, bool]:
    """Step 4: Kritzman emergency brake (unchanged)."""
    if turb_pctl > 0.95 and ar_z > 2.0:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                freed = weights[a] * 0.50
                weights[a] -= freed
                weights["cash"] = weights.get("cash", 0) + freed
        return weights, True
    return weights, False


def step5_normalize(weights: dict) -> dict:
    """Step 5: Enforce limits and normalize. Iterative cap-then-normalize."""
    w = dict(weights)

    # Single-eligible-asset cap: if only one risky asset has weight, cap at 40%
    risky_with_weight = [a for a in w if a != "cash" and w.get(a, 0) > 0.01]
    if len(risky_with_weight) == 1:
        a = risky_with_weight[0]
        if w[a] > 0.40:
            excess = w[a] - 0.40
            w[a] = 0.40
            w["cash"] = w.get("cash", 0) + excess

    for iteration in range(10):
        # Normalize to 1.0 first
        total = sum(max(v, 0) for v in w.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            w = {a: max(v, 0) / total for a, v in w.items()}

        changed = False

        # Cap single asset at 60%
        for a in w:
            if w[a] > 0.60:
                excess = w[a] - 0.60
                w[a] = 0.60
                w["cash"] = w.get("cash", 0) + excess
                changed = True

        # Cap equity at 85%
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > 0.85:
            r = 0.85 / eq
            for a in EQUITY:
                freed = w[a] - w[a] * r
                w[a] *= r
                w["cash"] = w.get("cash", 0) + freed
            changed = True

        # Floor cash at 3%
        if w.get("cash", 0) < 0.03:
            deficit = 0.03 - w["cash"]
            others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others:
                    w[a] -= deficit * (w[a] / tot)
            w["cash"] = 0.03
            changed = True

        if not changed:
            break

    # Final normalize
    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {a: max(v, 0) / total for a, v in w.items()}

    return w
