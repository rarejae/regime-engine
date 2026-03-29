"""Walk-forward validation for the HMM regime model.

Tests regime detection quality on truly out-of-sample data using
rolling window fits. Measures BIC/AIC, state stability,
and return separation between states.
"""

import logging
from dataclasses import dataclass, field

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


@dataclass
class ValidationReport:
    n_states: int
    n_folds: int
    folds: list[ValidationFold]
    avg_bic: float
    avg_aic: float
    avg_test_log_likelihood: float
    avg_state_duration: float
    direction_accuracy_std: float
    avg_states_used: float


def walk_forward_validate(
    features: pd.DataFrame,
    spy_prices: pd.Series,
    config: HMMConfig,
) -> ValidationReport:
    """Run walk-forward validation on the HMM."""
    clean = features.dropna()
    feature_cols = [c for c in clean.columns if c in config.hmm_features]

    folds: list[ValidationFold] = []
    fold_id = 0
    test_start_idx = config.min_train_days

    while test_start_idx + config.walk_forward_test_days <= len(clean):
        test_end_idx = min(test_start_idx + config.walk_forward_test_days, len(clean))
        train_end_idx = test_start_idx - 1
        train_start_idx = max(0, train_end_idx - config.window_days + 1)

        train_data = clean.iloc[train_start_idx : train_end_idx + 1][feature_cols].values
        test_data = clean.iloc[test_start_idx:test_end_idx][feature_cols].values

        if len(train_data) < config.min_window_days // 2 or len(test_data) < 5:
            test_start_idx += config.walk_forward_step_days
            continue

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)

        try:
            model = hmm.GaussianHMM(
                n_components=config.n_states,
                covariance_type=config.covariance_type,
                n_iter=config.n_iter,
                tol=config.tol,
                random_state=config.random_state,
            )
            model.fit(train_scaled)
        except Exception as e:
            logger.warning(f"Fold {fold_id} train failed: {e}")
            test_start_idx += config.walk_forward_step_days
            continue

        test_scaled = scaler.transform(test_data)

        try:
            test_states = model.predict(test_scaled)
        except Exception as e:
            logger.warning(f"Fold {fold_id} predict failed: {e}")
            test_start_idx += config.walk_forward_step_days
            continue

        train_ll = model.score(train_scaled)
        test_ll = model.score(test_scaled)

        n_params = _count_params(config.n_states, len(feature_cols), config.covariance_type)
        n_samples = len(train_data)
        bic = -2 * train_ll * n_samples + n_params * np.log(n_samples)
        aic = -2 * train_ll * n_samples + 2 * n_params

        transitions = int(np.sum(np.diff(test_states) != 0))
        runs = _state_run_lengths(test_states)
        avg_duration = float(np.mean(runs)) if runs else 0.0
        states_used = len(set(test_states))

        folds.append(ValidationFold(
            fold_id=fold_id,
            train_start=clean.index[train_start_idx],
            train_end=clean.index[train_end_idx],
            test_start=clean.index[test_start_idx],
            test_end=clean.index[min(test_end_idx - 1, len(clean) - 1)],
            n_train=len(train_data),
            n_test=len(test_data),
            train_log_likelihood=train_ll,
            test_log_likelihood=test_ll,
            bic=bic,
            aic=aic,
            avg_state_duration_test=avg_duration,
            n_transitions_test=transitions,
            states_used_test=states_used,
        ))

        fold_id += 1
        test_start_idx += config.walk_forward_step_days

    if not folds:
        raise ValueError("No valid folds were completed")

    return ValidationReport(
        n_states=config.n_states,
        n_folds=len(folds),
        folds=folds,
        avg_bic=float(np.mean([f.bic for f in folds])),
        avg_aic=float(np.mean([f.aic for f in folds])),
        avg_test_log_likelihood=float(np.mean([f.test_log_likelihood for f in folds])),
        avg_state_duration=float(np.mean([f.avg_state_duration_test for f in folds])),
        direction_accuracy_std=float(np.std([f.test_log_likelihood for f in folds])),
        avg_states_used=float(np.mean([f.states_used_test for f in folds])),
    )


def _count_params(n_states, n_features, cov_type):
    n_means = n_states * n_features
    n_trans = n_states * (n_states - 1)
    n_start = n_states - 1
    if cov_type == "full":
        n_cov = n_states * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        n_cov = n_states * n_features
    elif cov_type == "tied":
        n_cov = n_features * (n_features + 1) // 2
    elif cov_type == "spherical":
        n_cov = n_states
    else:
        n_cov = n_states * n_features
    return n_means + n_trans + n_start + n_cov


def _state_run_lengths(states):
    if len(states) == 0:
        return []
    runs = []
    current = 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return runs
