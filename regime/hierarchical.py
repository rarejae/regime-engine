"""Hierarchical signal system: Faber → Harvey → HMM → Kritzman."""

import numpy as np

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
EQUITY = ["IVV", "QQQ"]
ASSETS = list(BASELINE.keys())
MAX_SINGLE = 0.60
MAX_EQUITY = 0.85
MIN_CASH = 0.03


def step1_faber(trend_signals: dict) -> tuple[dict, float, list]:
    """Step 1: Faber gatekeeper — in or out per asset.

    Returns (eligible_weights, pool_size, ineligible_list).
    """
    w = {}
    pool = 0.0
    ineligible = []

    for a in ASSETS:
        if a == "cash":
            w[a] = BASELINE[a]
            continue
        if trend_signals.get(a, True):
            w[a] = BASELINE[a]  # eligible — start at baseline
        else:
            w[a] = 0.0
            pool += BASELINE[a]
            ineligible.append(a)

    return w, pool, ineligible


def step2_harvey(weights: dict, pool: float, harvey_er: dict) -> tuple[dict, dict]:
    """Step 2: Harvey directs freed capital to eligible assets with positive expected returns.

    Returns (updated_weights, allocation_detail).
    """
    detail = {"pool": pool, "directed_to": {}}

    if pool <= 0.001:
        return weights, detail

    # Eligible assets with positive expected return
    candidates = {}
    for a in ASSETS:
        if a == "cash":
            continue
        if weights.get(a, 0) > 0:  # eligible (passed Faber)
            er = harvey_er.get(a, 0)
            if er > 0:
                candidates[a] = er

    if not candidates:
        # No positive expected returns — all to cash
        weights["cash"] = weights.get("cash", 0) + pool
        detail["directed_to"]["cash"] = pool
        return weights, detail

    # Distribute proportionally to positive expected returns
    total_er = sum(candidates.values())
    for a, er in candidates.items():
        share = pool * (er / total_er)
        weights[a] = weights.get(a, 0) + share
        detail["directed_to"][a] = share

    return weights, detail


def step3_hmm(weights: dict, bull_prob: float) -> tuple[dict, str]:
    """Step 3: HMM conviction adjustment on equity.

    Returns (updated_weights, hmm_action).
    """
    if bull_prob > 0.7:
        # Boost equity by 15%, fund from cash
        action = "boost_15pct"
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                boost = weights[a] * 0.15
                weights[a] += boost
                weights["cash"] = max(weights.get("cash", 0) - boost, 0)
    elif bull_prob < 0.3:
        # Reduce equity by 15%, freed to cash
        action = "reduce_15pct"
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                reduction = weights[a] * 0.15
                weights[a] -= reduction
                weights["cash"] = weights.get("cash", 0) + reduction
    else:
        action = "no_change"

    return weights, action


def step4_kritzman(weights: dict, turb_pctl: float, ar_z: float) -> tuple[dict, bool]:
    """Step 4: Kritzman emergency brake. Only fires when BOTH extreme.

    Returns (updated_weights, fired).
    """
    if turb_pctl > 0.95 and ar_z > 2.0:
        # Cut all equity to 50% of current
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                freed = weights[a] * 0.50
                weights[a] -= freed
                weights["cash"] = weights.get("cash", 0) + freed
        return weights, True

    return weights, False


def step5_normalize(weights: dict) -> dict:
    """Step 5: Enforce limits and normalize."""
    w = dict(weights)

    for _ in range(5):
        # Cap single
        for a in w:
            if w[a] > MAX_SINGLE:
                excess = w[a] - MAX_SINGLE
                w[a] = MAX_SINGLE
                w["cash"] = w.get("cash", 0) + excess

        # Cap equity
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > MAX_EQUITY:
            ratio = MAX_EQUITY / eq
            for a in EQUITY:
                freed = w[a] - w[a] * ratio
                w[a] *= ratio
                w["cash"] = w.get("cash", 0) + freed

        # Floor cash
        if w.get("cash", 0) < MIN_CASH:
            deficit = MIN_CASH - w["cash"]
            others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others:
                    w[a] -= deficit * (w[a] / tot)
            w["cash"] = MIN_CASH

    total = sum(max(v, 0) for v in w.values())
    if total > 0:
        w = {a: max(v, 0) / total for a, v in w.items()}
    return w


def run_hierarchy(trend_signals, harvey_er, bull_prob, turb_pctl, ar_z):
    """Run full 4-step hierarchy. Returns (final_weights, trace)."""
    trace = {}

    # Step 1
    w, pool, ineligible = step1_faber(trend_signals)
    trace["step1"] = {"pool": pool, "ineligible": ineligible,
                       "weights": dict(w)}

    # Step 2
    w, harvey_detail = step2_harvey(w, pool, harvey_er)
    trace["step2"] = harvey_detail

    # Step 3
    w, hmm_action = step3_hmm(w, bull_prob)
    trace["step3"] = {"action": hmm_action, "bull_prob": bull_prob}

    # Step 4
    w, kritz_fired = step4_kritzman(w, turb_pctl, ar_z)
    trace["step4"] = {"fired": kritz_fired, "turb_pctl": turb_pctl, "ar_z": ar_z}

    # Step 5
    w = step5_normalize(w)
    trace["final"] = dict(w)

    return w, trace
