"""Sleeve notionals from V19d weights. Partial modes are 70% of the sleeve."""

from __future__ import annotations

WEIGHTS = {"pod1": 0.45, "pod2": 0.45, "gold": 0.10}
PARTIAL = {"qqq_partial", "ivv_partial"}


def sleeve_notional(capital: float, sleeve: str, mode: str | None) -> float:
    if not mode or mode == "cash":
        return 0.0
    w = WEIGHTS.get(sleeve, 0.0)
    if mode in PARTIAL:
        w *= 0.70
    return float(capital) * w


def shares_for(notional: float, px: float) -> float:
    if px <= 0 or notional <= 0:
        return 0.0
    return round(notional / px, 4)
