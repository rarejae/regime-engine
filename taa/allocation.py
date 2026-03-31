"""Two-step hierarchical allocation: Faber filter → Harvey capital direction."""

import numpy as np

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
EQUITY = ["IVV", "QQQ"]
ASSETS = list(BASELINE.keys())
MAX_SINGLE = 0.60
MAX_EQUITY = 0.85
MIN_CASH = 0.03


def direct_capital(weights: dict, pool: float,
                    expected_returns: dict, realized_vols: dict) -> tuple[dict, dict]:
    """Harvey inverse-vol capital direction for the reallocation pool.

    Distributes pool among eligible assets proportional to ER / vol.
    Single-eligible cap: max 40% to one asset.

    Returns (updated_weights, detail_dict).
    """
    detail = {"pool": pool, "scores": {}, "directed_to": {}}
    if pool <= 0.001:
        return weights, detail

    candidates = {}
    for a in ASSETS:
        if a == "cash" or weights.get(a, 0) <= 0:
            continue
        er = expected_returns.get(a, 0)
        vol = realized_vols.get(a, 0.15)
        if er > 0 and vol > 0.01:
            score = er / vol
            candidates[a] = score
            detail["scores"][a] = {"er": er, "vol": vol, "score": score}

    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool
        detail["directed_to"]["cash"] = pool
        return weights, detail

    # Single-eligible cap
    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = min(pool, 0.40 - weights.get(a, 0))
        alloc = max(alloc, 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc)
        detail["directed_to"][a] = alloc
        if pool - alloc > 0.001:
            detail["directed_to"]["cash"] = pool - alloc
        return weights, detail

    total_score = sum(candidates.values())
    for a, score in candidates.items():
        share = pool * (score / total_score)
        weights[a] = weights.get(a, 0) + share
        detail["directed_to"][a] = share

    return weights, detail


def normalize(weights: dict) -> dict:
    """Enforce limits: max single 60%, max equity 85%, min cash 3%. Iterative."""
    w = dict(weights)

    # Single-eligible-asset extra cap
    risky = [a for a in w if a != "cash" and w.get(a, 0) > 0.01]
    if len(risky) == 1 and w[risky[0]] > 0.40:
        excess = w[risky[0]] - 0.40
        w[risky[0]] = 0.40
        w["cash"] = w.get("cash", 0) + excess

    for _ in range(10):
        total = sum(max(v, 0) for v in w.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            w = {a: max(v, 0) / total for a, v in w.items()}

        changed = False
        for a in w:
            if w[a] > MAX_SINGLE:
                w["cash"] = w.get("cash", 0) + w[a] - MAX_SINGLE
                w[a] = MAX_SINGLE
                changed = True

        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > MAX_EQUITY:
            r = MAX_EQUITY / eq
            for a in EQUITY:
                freed = w[a] - w[a] * r
                w[a] *= r
                w["cash"] = w.get("cash", 0) + freed
            changed = True

        if w.get("cash", 0) < MIN_CASH:
            deficit = MIN_CASH - w["cash"]
            others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others:
                    w[a] -= deficit * (w[a] / tot)
            w["cash"] = MIN_CASH
            changed = True

        if not changed:
            break

    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {a: max(v, 0) / total for a, v in w.items()}
    return w
