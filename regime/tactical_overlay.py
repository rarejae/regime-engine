"""Tactical overlay: static baseline + rare high-conviction Harvey tilts."""

import numpy as np
import pandas as pd

BASELINE = {
    "IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05,
    "IAU": 0.10, "DBC": 0.05, "cash": 0.10,
}
EQUITY = ["IVV", "QQQ"]
DEFENSIVE = ["VGLT", "IAU", "DBC", "cash"]
MAX_SINGLE = 0.60
MAX_EQUITY = 0.90
MIN_CASH = 0.03
SHIFT_SIZE = 0.15  # 15% portfolio shift on signal


def compute_confidence(
    similar_fwd_returns: dict[str, list[float]],
    min_distance: float,
    trailing_min_dists: list[float],
    trailing_confidences: list[float],
) -> dict:
    """Compute composite confidence from Harvey similar months.

    Returns dict with confidence, confidence_pctl, per-asset metrics.
    """
    # Per-asset metrics
    asset_metrics = {}
    for asset, rets in similar_fwd_returns.items():
        rets = [r for r in rets if not np.isnan(r)]
        if len(rets) < 5:
            asset_metrics[asset] = {"signal_strength": 0, "agreement": 0.5, "mean_fwd": 0}
            continue
        mean_fwd = np.mean(rets)
        std_fwd = np.std(rets)
        signal_strength = mean_fwd / std_fwd if std_fwd > 0 else 0
        agreement = np.mean([1 if r * mean_fwd > 0 else 0 for r in rets])
        asset_metrics[asset] = {
            "signal_strength": signal_strength,
            "agreement": agreement,
            "mean_fwd": mean_fwd,
        }

    # Analog quality
    if trailing_min_dists and len(trailing_min_dists) >= 10:
        pctl = sum(1 for d in trailing_min_dists if d >= min_distance) / len(trailing_min_dists)
        analog_quality = pctl  # closer = higher quality (lower distance = higher percentile)
    else:
        analog_quality = 0.5

    # Composite equity confidence
    ivv = asset_metrics.get("IVV", {"signal_strength": 0, "agreement": 0.5})
    qqq = asset_metrics.get("QQQ", {"signal_strength": 0, "agreement": 0.5})

    equity_signal = ivv["signal_strength"] * 0.6 + qqq["signal_strength"] * 0.4
    equity_agreement = ivv["agreement"] * 0.6 + qqq["agreement"] * 0.4

    confidence = equity_signal * analog_quality * (equity_agreement - 0.5) * 2

    # Percentile rank against trailing history
    if trailing_confidences and len(trailing_confidences) >= 20:
        window = trailing_confidences[-120:] if len(trailing_confidences) > 120 else trailing_confidences
        confidence_pctl = sum(1 for c in window if c <= confidence) / len(window)
    else:
        confidence_pctl = 0.5

    return {
        "confidence": confidence,
        "confidence_pctl": confidence_pctl,
        "analog_quality": analog_quality,
        "equity_signal": equity_signal,
        "equity_agreement": equity_agreement,
        "asset_metrics": asset_metrics,
    }


def apply_overlay(
    confidence_pctl: float,
    asset_metrics: dict,
) -> dict:
    """Apply tactical overlay to baseline weights.

    Returns (weights, action, description).
    """
    w = dict(BASELINE)
    action = "hold"
    desc = ""

    if confidence_pctl > 0.90:
        # Strong bullish — shift 15% from defensive to equity
        action = "bullish"
        # Which equity has better signal?
        ivv_sig = asset_metrics.get("IVV", {}).get("signal_strength", 0)
        qqq_sig = asset_metrics.get("QQQ", {}).get("signal_strength", 0)
        total_sig = abs(ivv_sig) + abs(qqq_sig)
        if total_sig > 0:
            ivv_share = abs(ivv_sig) / total_sig
        else:
            ivv_share = 0.6

        # Reduce defensives proportionally
        def_total = sum(w[a] for a in DEFENSIVE)
        shift = min(SHIFT_SIZE, def_total * 0.8)  # Don't drain all defensives
        for a in DEFENSIVE:
            take = shift * (w[a] / def_total) if def_total > 0 else 0
            w[a] -= take

        w["IVV"] += shift * ivv_share
        w["QQQ"] += shift * (1 - ivv_share)
        desc = f"Bullish: +{shift:.0%} equity (IVV {ivv_share:.0%} / QQQ {1-ivv_share:.0%})"

    elif confidence_pctl < 0.10:
        # Strong bearish — shift 15% from equity to defensive
        action = "bearish"
        eq_total = sum(w[a] for a in EQUITY)
        shift = min(SHIFT_SIZE, eq_total * 0.5)  # Don't drain all equity

        for a in EQUITY:
            take = shift * (w[a] / eq_total) if eq_total > 0 else 0
            w[a] -= take

        # Best defensive asset from similar months
        best_def = "cash"
        best_fwd = -999
        for a in DEFENSIVE:
            fwd = asset_metrics.get(a, {}).get("mean_fwd", 0)
            if fwd > best_fwd:
                best_fwd = fwd
                best_def = a
        if best_fwd < 0:
            best_def = "cash"

        w[best_def] += shift
        desc = f"Bearish: -{shift:.0%} equity → {best_def}"

    # Enforce limits
    w = _enforce_limits(w)
    return w, action, desc


def _enforce_limits(w):
    for _ in range(5):
        for a in w:
            w[a] = min(w[a], MAX_SINGLE)
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > MAX_EQUITY:
            ratio = MAX_EQUITY / eq
            for a in EQUITY: w[a] *= ratio
        if w.get("cash", 0) < MIN_CASH:
            deficit = MIN_CASH - w["cash"]
            others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others: w[a] -= deficit * (w[a] / tot)
            w["cash"] = MIN_CASH

    total = sum(max(v, 0) for v in w.values())
    if total > 0:
        w = {a: max(v, 0) / total for a, v in w.items()}
    return w
