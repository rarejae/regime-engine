"""Regime analysis: compare HMM to rules-based, generate timeline."""

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
    """Compare HMM regime detection against rules-based classifier."""
    rules_rows = []
    for dt, cv in rules_cvs.items():
        rules_rows.append({
            "trade_date": pd.Timestamp(dt),
            "regime_label": cv.regime_label if hasattr(cv, "regime_label") else "MIXED",
        })
    rules_df = pd.DataFrame(rules_rows).set_index("trade_date")
    hmm_pred = hmm_predictions.set_index("trade_date")
    merged = hmm_pred.join(rules_df, how="inner")

    hmm_dir_map = {p.state_id: p.direction for p in state_profiles}
    rules_dir_map = {
        "GOLDILOCKS": "bullish", "REFLATION": "bullish",
        "STAGFLATION": "bearish", "DEFLATION": "bearish",
        "MIXED": "neutral",
    }

    fwd_1m = spy_prices.pct_change(21).shift(-21)
    hmm_c = rules_c = agree = total = 0
    disag = []

    for dt, row in merged.iterrows():
        h = hmm_dir_map.get(row["most_likely_state"], "neutral")
        r = rules_dir_map.get(row.get("regime_label", "MIXED"), "neutral")
        ret = fwd_1m.get(dt)
        if ret is None or np.isnan(ret) or h == "neutral" or r == "neutral":
            continue
        total += 1
        actual = "bullish" if ret > 0 else "bearish"
        if h == r:
            agree += 1
        else:
            disag.append({"date": dt, "hmm": h, "rules": r, "actual": actual,
                         "ret": ret, "hmm_right": h == actual, "rules_right": r == actual})
        if h == actual:
            hmm_c += 1
        if r == actual:
            rules_c += 1

    return {
        "total_comparable": total,
        "agreement_rate": agree / total if total else 0,
        "hmm_accuracy_1m": hmm_c / total if total else 0,
        "rules_accuracy_1m": rules_c / total if total else 0,
        "hmm_advantage": (hmm_c - rules_c) / total if total else 0,
        "disagreements": pd.DataFrame(disag),
    }


def generate_regime_timeline(predictions, state_profiles, spy_prices):
    """Generate monthly regime timeline."""
    pred = predictions.set_index("trade_date")
    monthly = pred.resample("MS").first().dropna(subset=["most_likely_state"])
    fwd_1m = spy_prices.pct_change(21).shift(-21)
    fwd_3m = spy_prices.pct_change(63).shift(-63)
    lmap = {p.state_id: p.label for p in state_profiles}
    dmap = {p.state_id: p.direction for p in state_profiles}

    rows = []
    for dt, row in monthly.iterrows():
        s = int(row["most_likely_state"])
        rows.append({"date": dt, "state": s, "label": lmap.get(s, f"s{s}"),
                     "confidence": row["confidence"], "direction": dmap.get(s, "neutral"),
                     "spy_1m": fwd_1m.get(dt, np.nan), "spy_3m": fwd_3m.get(dt, np.nan)})
    return pd.DataFrame(rows)
