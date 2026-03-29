"""Post-hoc state labeling.

States are discovered by the HMM, then labeled by examining their
emission distributions and forward return characteristics.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hmm.config import HMMConfig

logger = logging.getLogger(__name__)


@dataclass
class StateProfile:
    """Profile of a single HMM state."""
    state_id: int
    label: str

    pct_of_time: float
    avg_duration_days: float

    avg_spy_return_1m: float
    avg_spy_return_3m: float
    return_std_1m: float
    direction: str
    direction_accuracy_1m: float

    feature_means: dict
    recommended_strategy: str
    example_periods: list[str] = field(default_factory=list)


def label_states(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    spy_prices: pd.Series,
    config: HMMConfig,
) -> list[StateProfile]:
    """Analyze each HMM state and assign labels."""
    pred = predictions.set_index("trade_date")

    fwd_1m = spy_prices.pct_change(21).shift(-21)
    fwd_3m = spy_prices.pct_change(63).shift(-63)

    aligned = pred.join(features, how="left")
    aligned["spy_fwd_1m"] = fwd_1m.reindex(aligned.index)
    aligned["spy_fwd_3m"] = fwd_3m.reindex(aligned.index)

    profiles = []

    for state_id in range(config.n_states):
        state_mask = aligned["most_likely_state"] == state_id
        state_data = aligned[state_mask]

        if len(state_data) == 0:
            logger.warning(f"State {state_id} has no observations")
            continue

        pct_time = len(state_data) / len(aligned)
        runs = _compute_runs(aligned["most_likely_state"], state_id)
        avg_duration = np.mean(runs) if runs else 0

        r1 = state_data["spy_fwd_1m"].dropna()
        r3 = state_data["spy_fwd_3m"].dropna()
        avg_ret_1m = r1.mean() if len(r1) else 0.0
        avg_ret_3m = r3.mean() if len(r3) else 0.0
        ret_std_1m = r1.std() if len(r1) > 1 else 0.0

        if avg_ret_1m > 0.005:
            direction = "bullish"
        elif avg_ret_1m < -0.005:
            direction = "bearish"
        else:
            direction = "neutral"

        if direction == "bullish":
            correct = (r1 > 0).sum()
        elif direction == "bearish":
            correct = (r1 < 0).sum()
        else:
            correct = len(r1)
        dir_accuracy = correct / len(r1) if len(r1) > 0 else 0

        feature_cols = [c for c in config.hmm_features if c in state_data.columns]
        feature_means = {col: float(state_data[col].mean()) for col in feature_cols}

        label = _auto_label(avg_ret_1m, ret_std_1m, feature_means, pct_time)
        strategy = _recommend_strategy(direction, feature_means, ret_std_1m)
        examples = _find_example_periods(aligned["most_likely_state"], state_id, aligned.index)

        profiles.append(StateProfile(
            state_id=state_id,
            label=label,
            pct_of_time=pct_time,
            avg_duration_days=avg_duration,
            avg_spy_return_1m=avg_ret_1m,
            avg_spy_return_3m=avg_ret_3m,
            return_std_1m=ret_std_1m,
            direction=direction,
            direction_accuracy_1m=dir_accuracy,
            feature_means=feature_means,
            recommended_strategy=strategy,
            example_periods=examples,
        ))

    profiles.sort(key=lambda p: p.pct_of_time, reverse=True)
    return profiles


def _compute_runs(state_series: pd.Series, target_state: int) -> list[int]:
    runs = []
    current_run = 0
    for val in state_series:
        if val == target_state:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    return runs


def _auto_label(avg_ret, ret_std, feature_means, pct_time):
    vix_z = feature_means.get("vix_z", 0)
    inflation_z = feature_means.get("inflation_z", 0)

    if avg_ret > 0.02 and ret_std > 0.05:
        return "recovery_snap"
    elif avg_ret > 0.005 and vix_z < 0.5:
        return "steady_bull"
    elif avg_ret < -0.01 and vix_z > 1.0:
        return "stress_panic"
    elif avg_ret < -0.005 and vix_z > 0:
        return "risk_off"
    elif inflation_z > 0.5 and abs(avg_ret) < 0.01:
        return "inflation_grind"
    elif pct_time > 0.30 and abs(avg_ret) < 0.005:
        return "neutral_chop"
    elif avg_ret > 0.005:
        return "quiet_bull"
    else:
        return "mixed_drift"


def _recommend_strategy(direction, feature_means, ret_std):
    vix_z = feature_means.get("vix_z", 0)
    if direction == "bullish" and vix_z > 0.5:
        return "bull_put_spread"
    elif direction == "bullish":
        return "iron_condor"
    elif direction == "bearish" and vix_z > 0:
        return "bear_call_spread"
    else:
        return "iron_condor"


def _find_example_periods(state_series, target_state, index, top_n=3):
    runs = []
    start = None
    for i, val in enumerate(state_series):
        if val == target_state:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1, i - start))
                start = None
    if start is not None:
        runs.append((start, len(state_series) - 1, len(state_series) - start))

    runs.sort(key=lambda x: x[2], reverse=True)
    examples = []
    for start_idx, end_idx, length in runs[:top_n]:
        d1 = index[start_idx].strftime("%Y-%m-%d")
        d2 = index[min(end_idx, len(index) - 1)].strftime("%Y-%m-%d")
        examples.append(f"{d1} to {d2} ({length}d)")
    return examples
