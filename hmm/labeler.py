"""Post-hoc state labeling with relative direction classification.

v3.1 fixes:
- Direction classified by RANKING states relative to each other, not by
  comparing to a fixed threshold (which fails due to equity risk premium).
- Excess return = state avg return minus unconditional mean.
- Direction accuracy computed on excess returns.
- Confidence calibration reported.
- Transition matrix self-transition probabilities reported.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hmm.config import HMMConfig

logger = logging.getLogger(__name__)


@dataclass
class StateProfile:
    state_id: int
    label: str
    pct_of_time: float
    avg_duration_days: float
    median_duration_days: float
    model_implied_duration: float   # From transition matrix self-transition prob
    self_transition_prob: float     # P(stay in state)

    avg_spy_return_1m: float
    avg_spy_return_3m: float
    excess_return_1m: float         # avg - unconditional mean
    return_std_1m: float
    direction: str                  # Relative: bearish/neutral/bullish
    direction_accuracy_1m: float    # Accuracy using excess returns

    avg_return_at_entry: float
    avg_return_late: float
    entry_direction: str
    late_direction: str

    feature_means: dict
    recommended_strategy: str
    example_periods: list[str] = field(default_factory=list)


@dataclass
class LabelingResult:
    """Full labeling output including metadata."""
    profiles: list[StateProfile]
    unconditional_mean_1m: float
    confidence_median: float
    confidence_p25: float
    confidence_p10: float
    transition_matrix: np.ndarray | None


def label_states(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    spy_prices: pd.Series,
    config: HMMConfig,
    last_model=None,
) -> LabelingResult:
    """Analyze each HMM state with relative direction classification."""
    pred = predictions.copy().set_index("trade_date")

    fwd_1m = spy_prices.pct_change(21).shift(-21)
    fwd_3m = spy_prices.pct_change(63).shift(-63)

    aligned = pred.join(features, how="left")
    aligned["spy_fwd_1m"] = fwd_1m.reindex(aligned.index)
    aligned["spy_fwd_3m"] = fwd_3m.reindex(aligned.index)
    aligned["days_in_state"] = _compute_days_in_state(aligned["most_likely_state"])

    # Unconditional mean return (equity risk premium baseline)
    unconditional_mean = aligned["spy_fwd_1m"].mean()
    logger.info(f"Unconditional mean 1m return: {unconditional_mean:.4f} ({unconditional_mean:.2%})")

    # Confidence calibration
    conf = pred["confidence"]
    conf_median = float(conf.median())
    conf_p25 = float(conf.quantile(0.25))
    conf_p10 = float(conf.quantile(0.10))
    if conf_median > 0.95:
        logger.warning(f"Confidence median={conf_median:.3f} > 0.95 — probability vectors lack gradation")

    # Transition matrix from the last fitted model
    trans_matrix = None
    if last_model is not None and hasattr(last_model, "transmat_"):
        trans_matrix = last_model.transmat_

    # First pass: compute raw stats per state
    raw_stats = {}
    for state_id in range(config.n_states):
        mask = aligned["most_likely_state"] == state_id
        sd = aligned[mask]
        if len(sd) == 0:
            continue

        r1 = sd["spy_fwd_1m"].dropna()
        raw_stats[state_id] = {
            "data": sd,
            "r1": r1,
            "r3": sd["spy_fwd_3m"].dropna(),
            "avg1": r1.mean() if len(r1) else 0.0,
            "pct_time": len(sd) / len(aligned),
        }

    # Second pass: classify direction by RANKING states
    state_ids = sorted(raw_stats.keys())
    sorted_by_return = sorted(state_ids, key=lambda s: raw_stats[s]["avg1"])

    # With 4 states: lowest = bearish, highest = bullish, middle 2 = neutral
    # With 3 states: lowest = bearish, highest = bullish, middle = neutral
    n = len(sorted_by_return)
    direction_map = {}
    for rank, sid in enumerate(sorted_by_return):
        if rank == 0:
            direction_map[sid] = "bearish"
        elif rank == n - 1:
            direction_map[sid] = "bullish"
        else:
            direction_map[sid] = "neutral"

    logger.info(f"Direction ranking (by avg 1m return):")
    for sid in sorted_by_return:
        excess = raw_stats[sid]["avg1"] - unconditional_mean
        logger.info(f"  State {sid}: avg={raw_stats[sid]['avg1']:.4f}, "
                   f"excess={excess:+.4f} → {direction_map[sid]}")

    # Third pass: build full profiles
    profiles = []
    for state_id in state_ids:
        stats = raw_stats[state_id]
        sd = stats["data"]
        r1 = stats["r1"]
        r3 = stats["r3"]
        avg1 = stats["avg1"]
        avg3 = r3.mean() if len(r3) else 0.0
        std1 = r1.std() if len(r1) > 1 else 0.0
        excess = avg1 - unconditional_mean

        direction = direction_map[state_id]

        # Direction accuracy using EXCESS returns
        dir_acc = _direction_accuracy_excess(r1, direction, unconditional_mean)

        # Duration
        runs = _compute_runs(aligned["most_likely_state"], state_id)
        avg_dur = np.mean(runs) if runs else 0
        med_dur = np.median(runs) if runs else 0

        # Model-implied duration from transition matrix
        self_trans = 0.0
        implied_dur = 0.0
        if trans_matrix is not None and state_id < trans_matrix.shape[0]:
            self_trans = float(trans_matrix[state_id, state_id])
            implied_dur = 1.0 / (1.0 - self_trans) if self_trans < 1.0 else float("inf")
            if self_trans < 0.90:
                logger.warning(f"State {state_id}: self-transition={self_trans:.3f} < 0.90 — UNSTABLE")

        # Entry vs late
        entry = sd[sd["days_in_state"] <= 5]["spy_fwd_1m"].dropna()
        late = sd[sd["days_in_state"] >= 10]["spy_fwd_1m"].dropna()
        ret_entry = entry.mean() if len(entry) else np.nan
        ret_late = late.mean() if len(late) else np.nan

        # Entry/late direction using excess returns
        entry_dir = _classify_excess(ret_entry, unconditional_mean)
        late_dir = _classify_excess(ret_late, unconditional_mean)

        if entry_dir != late_dir:
            logger.info(f"State {state_id}: ENTRY/LATE INVERSION — "
                       f"entry={entry_dir}({ret_entry:.3f}), late={late_dir}({ret_late:.3f})")

        fcols = [c for c in config.hmm_features if c in sd.columns]
        fmeans = {c: float(sd[c].mean()) for c in fcols}

        label = _auto_label(excess, std1, fmeans, stats["pct_time"], avg_dur,
                           direction, entry_dir, late_dir)
        strat = _recommend_strategy(direction, fmeans, std1, entry_dir, late_dir)
        examples = _find_example_periods(aligned["most_likely_state"], state_id, aligned.index)

        profiles.append(StateProfile(
            state_id=state_id, label=label,
            pct_of_time=stats["pct_time"],
            avg_duration_days=avg_dur, median_duration_days=med_dur,
            model_implied_duration=implied_dur, self_transition_prob=self_trans,
            avg_spy_return_1m=avg1, avg_spy_return_3m=avg3,
            excess_return_1m=excess, return_std_1m=std1,
            direction=direction, direction_accuracy_1m=dir_acc,
            avg_return_at_entry=ret_entry, avg_return_late=ret_late,
            entry_direction=entry_dir, late_direction=late_dir,
            feature_means=fmeans, recommended_strategy=strat, example_periods=examples,
        ))

    profiles.sort(key=lambda p: p.avg_spy_return_1m)  # Sort by return: bearish first

    return LabelingResult(
        profiles=profiles,
        unconditional_mean_1m=unconditional_mean,
        confidence_median=conf_median,
        confidence_p25=conf_p25,
        confidence_p10=conf_p10,
        transition_matrix=trans_matrix,
    )


def _direction_accuracy_excess(returns: pd.Series, direction: str,
                                unconditional_mean: float) -> float:
    """Accuracy using excess returns (return - unconditional mean)."""
    valid = returns.dropna()
    if len(valid) == 0:
        return 0.0
    excess = valid - unconditional_mean
    if direction == "bullish":
        return float((excess > 0).mean())
    elif direction == "bearish":
        return float((excess < 0).mean())
    return float((excess.abs() < excess.std()).mean()) if len(excess) > 1 else 0.5


def _classify_excess(avg_ret: float, unconditional_mean: float,
                     threshold: float = 0.002) -> str:
    """Classify direction using excess return over unconditional mean."""
    if np.isnan(avg_ret):
        return "neutral"
    excess = avg_ret - unconditional_mean
    if excess > threshold:
        return "bullish"
    elif excess < -threshold:
        return "bearish"
    return "neutral"


def _auto_label(excess_ret, ret_std, fmeans, pct_time, avg_dur,
                direction, entry_dir, late_dir):
    nfci_z = fmeans.get("nfci_z", 0)
    vvix_z = fmeans.get("vvix_z", 0)
    wti_z = fmeans.get("wti_z", 0)

    if entry_dir == "bearish" and late_dir == "bullish":
        return "stress_recovery"
    if entry_dir == "bullish" and late_dir == "bearish":
        return "exhaustion_top"
    if direction == "bearish" and nfci_z > 0.3:
        return "stress"
    if direction == "bearish":
        return "risk_off"
    if direction == "bullish" and excess_ret > 0.003:
        return "strong_bull"
    if direction == "bullish":
        return "steady_growth"
    if abs(excess_ret) < 0.001 and pct_time > 0.20:
        return "neutral_drift"
    return "transition"


def _recommend_strategy(direction, fmeans, ret_std, entry_dir, late_dir):
    nfci_z = fmeans.get("nfci_z", 0)
    vvix_z = fmeans.get("vvix_z", 0)
    if entry_dir != late_dir:
        if entry_dir == "bearish":
            return "wait_then_bull_put_spread"
        return "take_profit_reduce_risk"
    if direction == "bullish" and nfci_z > 0.2:
        return "bull_put_spread"
    if direction == "bullish":
        return "iron_condor"
    if direction == "bearish":
        return "bear_call_spread"
    return "iron_condor"


def _compute_days_in_state(state_series):
    days = pd.Series(0, index=state_series.index, dtype=int)
    current = 1; prev = None
    for i, (idx, state) in enumerate(state_series.items()):
        if state == prev: current += 1
        else: current = 1
        days.iloc[i] = current; prev = state
    return days


def _compute_runs(state_series, target):
    runs, cur = [], 0
    for v in state_series:
        if v == target: cur += 1
        else:
            if cur > 0: runs.append(cur)
            cur = 0
    if cur > 0: runs.append(cur)
    return runs


def _find_example_periods(state_series, target, index, top_n=3):
    runs, start = [], None
    for i, v in enumerate(state_series):
        if v == target:
            if start is None: start = i
        else:
            if start is not None: runs.append((start, i - 1, i - start)); start = None
    if start is not None: runs.append((start, len(state_series) - 1, len(state_series) - start))
    runs.sort(key=lambda x: x[2], reverse=True)
    out = []
    for s, e, l in runs[:top_n]:
        d1 = index[s].strftime("%Y-%m-%d")
        d2 = index[min(e, len(index) - 1)].strftime("%Y-%m-%d")
        out.append(f"{d1} to {d2} ({l}d)")
    return out
