"""Blend Harvey monthly macro allocation with HMM daily trend signal."""

import numpy as np

EQUITY = ["IVV", "QQQ"]
DEFENSIVE = ["VGLT", "IAU", "DBC", "VNQ"]
MAX_SINGLE = 0.50
MAX_EQUITY = 0.70
MIN_CASH = 0.02


def blend_weights(
    harvey_w: dict,
    hmm_zone: str,
    bull_prob: float,
) -> dict:
    """Apply HMM trend signal to Harvey base allocation.

    Args:
        harvey_w: Harvey similarity-weighted allocation.
        hmm_zone: "bull", "bear", or "neutral" (after persistence filter).
        bull_prob: Raw bull probability from HMM.

    Returns:
        Blended weights dict.
    """
    w = dict(harvey_w)
    harvey_eq = sum(w.get(a, 0) for a in EQUITY)

    if hmm_zone == "bull":
        # HMM confirms uptrend — ensure adequate equity
        if harvey_eq < 0.40:
            target_eq = max(harvey_eq, 0.40)
            target_eq = min(target_eq, MAX_EQUITY)
            increase = target_eq - harvey_eq

            # Split increase between IVV and QQQ proportionally
            if harvey_eq > 0:
                ivv_share = w.get("IVV", 0) / harvey_eq
                qqq_share = w.get("QQQ", 0) / harvey_eq
            else:
                ivv_share = 0.67
                qqq_share = 0.33

            w["IVV"] = w.get("IVV", 0) + increase * ivv_share
            w["QQQ"] = w.get("QQQ", 0) + increase * qqq_share

            # Fund from cash first
            cash_available = max(w.get("cash", 0) - MIN_CASH, 0)
            from_cash = min(increase, cash_available)
            w["cash"] = w.get("cash", 0) - from_cash
            remaining = increase - from_cash

            # Then from defensives proportionally
            if remaining > 0:
                def_total = sum(w.get(a, 0) for a in DEFENSIVE)
                if def_total > 0:
                    for a in DEFENSIVE:
                        take = remaining * (w.get(a, 0) / def_total)
                        w[a] = max(w.get(a, 0) - take, 0)

    elif hmm_zone == "bear":
        # HMM detects trend break — reduce equity by 30%
        reduction = harvey_eq * 0.30
        if harvey_eq > 0:
            ivv_share = w.get("IVV", 0) / harvey_eq
            qqq_share = w.get("QQQ", 0) / harvey_eq
        else:
            ivv_share = qqq_share = 0

        w["IVV"] = max(w.get("IVV", 0) - reduction * ivv_share, 0)
        w["QQQ"] = max(w.get("QQQ", 0) - reduction * qqq_share, 0)
        w["cash"] = w.get("cash", 0) + reduction

    # else: neutral — no modification

    # Enforce hard limits
    w = _enforce_limits(w)
    return w


def _enforce_limits(w: dict) -> dict:
    """Enforce max single ETF, max equity, min cash, normalize."""
    all_assets = list(w.keys())

    for _ in range(5):
        # Cap single ETF
        for a in all_assets:
            if w.get(a, 0) > MAX_SINGLE:
                excess = w[a] - MAX_SINGLE
                w[a] = MAX_SINGLE
                w["cash"] = w.get("cash", 0) + excess

        # Cap total equity
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > MAX_EQUITY:
            ratio = MAX_EQUITY / eq
            freed = 0
            for a in EQUITY:
                old = w.get(a, 0)
                w[a] = old * ratio
                freed += old - w[a]
            w["cash"] = w.get("cash", 0) + freed

        # Floor cash
        if w.get("cash", 0) < MIN_CASH:
            deficit = MIN_CASH - w.get("cash", 0)
            others = [a for a in all_assets if a != "cash" and w.get(a, 0) > 0]
            total = sum(w[a] for a in others)
            if total > deficit:
                for a in others:
                    w[a] -= deficit * (w[a] / total)
                w["cash"] = MIN_CASH

    # Normalize to 1.0
    total = sum(max(v, 0) for v in w.values())
    if total > 0:
        w = {a: max(v, 0) / total for a, v in w.items()}

    return w
