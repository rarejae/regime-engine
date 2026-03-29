"""Walk-forward validation for the HMM regime model.

Tests regime detection quality on truly out-of-sample data.
Measures state separation, persistence, and information content.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from hmm.config import HMMConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    train_log_likelihood: float
    test_log_likelihood: float
    bic: float
    aic: float
    avg_state_duration_test: float
    n_transitions_test: int
    states_used_test: int
    state_return_separation: float


@dataclass
class ValidationReport:
    n_states: int
    n_folds: int
    folds: list
    avg_bic: float
    avg_aic: float
    avg_test_log_likelihood: float
    avg_state_duration: float
    avg_states_used: float
    avg_return_separation: float
    test_ll_std: float
    duration_std: float


def walk_forward_validate(
    features: pd.DataFrame,
    spy_prices: pd.Series,
    config: HMMConfig,
) -> ValidationReport:
    """Run walk-forward validation."""
    clean = features[config.hmm_features].dropna()
    fwd_1m = spy_prices.pct_change(21).shift(-21) if spy_prices is not None else None

    folds: list[ValidationFold] = []
    fold_id = 0
    idx = config.min_train_days

    while idx + config.walk_forward_test_days <= len(clean):
        te = min(idx + config.walk_forward_test_days, len(clean))
        tr_end = idx - 1
        tr_start = max(0, tr_end - config.window_days + 1)

        train = clean.iloc[tr_start:tr_end + 1].values
        test = clean.iloc[idx:te].values

        if len(train) < config.min_window_days // 2 or len(test) < 5:
            idx += config.walk_forward_step_days
            continue

        scaler = StandardScaler()
        train_s = scaler.fit_transform(train)

        try:
            model = hmm.GaussianHMM(
                n_components=config.n_states,
                covariance_type=config.covariance_type,
                n_iter=config.n_iter, tol=config.tol,
                random_state=config.random_state,
            )
            model.fit(train_s)
        except Exception:
            idx += config.walk_forward_step_days
            continue

        test_s = scaler.transform(test)
        try:
            test_states = model.predict(test_s)
        except Exception:
            idx += config.walk_forward_step_days
            continue

        train_ll = model.score(train_s)
        test_ll = model.score(test_s)
        n_p = _count_params(config.n_states, len(config.hmm_features), config.covariance_type)
        bic = -2 * train_ll * len(train) + n_p * np.log(len(train))
        aic = -2 * train_ll * len(train) + 2 * n_p

        trans = int(np.sum(np.diff(test_states) != 0))
        runs = _runs(test_states)
        avg_d = float(np.mean(runs)) if runs else 0
        n_used = len(set(test_states))

        sep = 0.0
        if fwd_1m is not None:
            dates = clean.index[idx:te]
            sep = _return_sep(test_states, dates, fwd_1m, config.n_states)

        folds.append(ValidationFold(
            fold_id=fold_id,
            train_start=clean.index[tr_start], train_end=clean.index[tr_end],
            test_start=clean.index[idx], test_end=clean.index[min(te - 1, len(clean) - 1)],
            n_train=len(train), n_test=len(test),
            train_log_likelihood=train_ll, test_log_likelihood=test_ll,
            bic=bic, aic=aic,
            avg_state_duration_test=avg_d, n_transitions_test=trans,
            states_used_test=n_used, state_return_separation=sep,
        ))
        fold_id += 1
        idx += config.walk_forward_step_days

    if not folds:
        raise ValueError("No valid folds completed")

    return ValidationReport(
        n_states=config.n_states, n_folds=len(folds), folds=folds,
        avg_bic=np.mean([f.bic for f in folds]),
        avg_aic=np.mean([f.aic for f in folds]),
        avg_test_log_likelihood=np.mean([f.test_log_likelihood for f in folds]),
        avg_state_duration=np.mean([f.avg_state_duration_test for f in folds]),
        avg_states_used=np.mean([f.states_used_test for f in folds]),
        avg_return_separation=np.mean([f.state_return_separation for f in folds]),
        test_ll_std=np.std([f.test_log_likelihood for f in folds]),
        duration_std=np.std([f.avg_state_duration_test for f in folds]),
    )


def _count_params(n_s, n_f, cov):
    n_m = n_s * n_f
    n_t = n_s * (n_s - 1)
    n_st = n_s - 1
    if cov == "full":
        n_c = n_s * n_f * (n_f + 1) // 2
    elif cov == "diag":
        n_c = n_s * n_f
    elif cov == "tied":
        n_c = n_f * (n_f + 1) // 2
    else:
        n_c = n_s * n_f
    return n_m + n_t + n_st + n_c


def _return_sep(states, dates, fwd, n_states):
    sr = {}
    for s, d in zip(states, dates):
        r = fwd.get(d)
        if r is not None and not np.isnan(r):
            sr.setdefault(s, []).append(r)
    if len(sr) < 2:
        return 0.0
    means = [np.mean(v) for v in sr.values() if len(v) > 5]
    stds = [np.std(v) for v in sr.values() if len(v) > 5]
    if len(means) < 2 or np.mean(stds) == 0:
        return 0.0
    return (max(means) - min(means)) / np.mean(stds)


def _runs(states):
    if len(states) == 0:
        return []
    out, c = [], 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            c += 1
        else:
            out.append(c)
            c = 1
    out.append(c)
    return out
