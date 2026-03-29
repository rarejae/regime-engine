"""Regime analysis tools — compare HMM to rules-based, generate timelines."""

import logging

import numpy as np
import pandas as pd

from hmm.config import HMMConfig
from hmm.labeler import StateProfile

logger = logging.getLogger(__name__)


def compare_to_rules_based(
    hmm_predictions: pd.DataFrame,
    rules_cvs: dict,
    spy_prices: pd.Series,
    state_profiles: list[StateProfile],
) -> dict:
    """Compare HMM regime detection against the rules-based classifier.

    Args:
        hmm_predictions: Output from fit_and_predict_rolling().
        rules_cvs: Dict mapping date → ConditionVector from the rules-based engine.
        spy_prices: SPY close prices.
        state_profiles: Labeled HMM state profiles.

    Returns:
        Dict with agreement rate, accuracy comparison, disagreement analysis.
    """
    hmm_direction_map = {p.state_id: p.direction for p in state_profiles}
    rules_direction_map = {
        "GOLDILOCKS": "bullish", "REFLATION": "bullish",
        "STAGFLATION": "bearish", "DEFLATION": "bearish",
        "MIXED": "neutral",
    }

    fwd_1m = spy_prices.pct_change(21).shift(-21)

    hmm_correct = rules_correct = agreement = total = 0
    disagreements = []

    for _, row in hmm_predictions.iterrows():
        dt = row["trade_date"]
        dt_date = dt.date() if hasattr(dt, "date") else dt

        hmm_dir = hmm_direction_map.get(row["most_likely_state"], "neutral")

        cv = rules_cvs.get(dt_date)
        if cv is None:
            continue
        rules_dir = rules_direction_map.get(cv.regime_label, "neutral")

        ret = fwd_1m.get(dt)
        if ret is None or np.isnan(ret):
            continue
        if hmm_dir == "neutral" or rules_dir == "neutral":
            continue

        total += 1
        actual_dir = "bullish" if ret > 0 else "bearish"

        if hmm_dir == rules_dir:
            agreement += 1
        else:
            disagreements.append({
                "date": dt,
                "hmm_said": hmm_dir,
                "rules_said": rules_dir,
                "actual": actual_dir,
                "spy_return": ret,
                "hmm_right": hmm_dir == actual_dir,
                "rules_right": rules_dir == actual_dir,
            })

        if hmm_dir == actual_dir:
            hmm_correct += 1
        if rules_dir == actual_dir:
            rules_correct += 1

    return {
        "total_comparable": total,
        "agreement_rate": agreement / total if total > 0 else 0,
        "hmm_accuracy_1m": hmm_correct / total if total > 0 else 0,
        "rules_accuracy_1m": rules_correct / total if total > 0 else 0,
        "hmm_advantage": (hmm_correct - rules_correct) / total if total > 0 else 0,
        "disagreements": pd.DataFrame(disagreements) if disagreements else pd.DataFrame(),
    }


def generate_regime_timeline(
    predictions: pd.DataFrame,
    state_profiles: list[StateProfile],
    spy_prices: pd.Series,
) -> pd.DataFrame:
    """Generate monthly regime timeline using HMM states."""
    pred = predictions.set_index("trade_date")
    monthly = pred.resample("MS").first().dropna(subset=["most_likely_state"])

    fwd_1m = spy_prices.pct_change(21).shift(-21)
    fwd_3m = spy_prices.pct_change(63).shift(-63)

    label_map = {p.state_id: p.label for p in state_profiles}
    direction_map = {p.state_id: p.direction for p in state_profiles}

    rows = []
    for dt, row in monthly.iterrows():
        state = int(row["most_likely_state"])
        rows.append({
            "date": dt,
            "state": state,
            "label": label_map.get(state, f"state_{state}"),
            "confidence": row["confidence"],
            "direction": direction_map.get(state, "neutral"),
            "spy_1m_fwd": fwd_1m.get(dt, np.nan),
            "spy_3m_fwd": fwd_3m.get(dt, np.nan),
        })

    return pd.DataFrame(rows)
