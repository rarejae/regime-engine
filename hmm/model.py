"""Core HMM model: fitting, prediction, rolling window management.

Uses hmmlearn GaussianHMM with multivariate observations.
Rolling window refitting ensures adaptation to regime changes.
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
class HMMResult:
    """Result of HMM fitting and prediction for a single date."""
    trade_date: pd.Timestamp
    state_probabilities: np.ndarray   # shape (n_states,)
    most_likely_state: int
    confidence: float
    model_log_likelihood: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp          # always < trade_date for PIT


def fit_and_predict_rolling(
    features: pd.DataFrame,
    config: HMMConfig,
) -> pd.DataFrame:
    """Fit HMM on rolling windows and produce state probabilities for each date.

    For each prediction date t:
    1. Training window = [t - window_days, t - 1] (PIT safe)
    2. Fit GaussianHMM on the training window
    3. Predict state probabilities from the full training sequence
    4. Output state probabilities aligned to date t
    """
    clean = features.dropna()
    feature_cols = [c for c in clean.columns if c in config.hmm_features]

    if not feature_cols:
        raise ValueError(f"No matching features found. Available: {list(clean.columns)}")

    logger.info(
        f"Fitting HMM with {config.n_states} states on {len(feature_cols)} features, "
        f"window={config.window_days}d, refit every {config.refit_frequency}d"
    )

    results: list[HMMResult] = []
    last_fit_idx = -config.refit_frequency
    current_model = None
    current_scaler = None

    for i in range(config.min_window_days, len(clean)):
        if (i - last_fit_idx) >= config.refit_frequency or current_model is None:
            train_end = i - 1
            train_start = max(0, train_end - config.window_days + 1)
            train_data = clean.iloc[train_start : train_end + 1][feature_cols].values

            if len(train_data) < config.min_window_days // 2:
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
                current_model = model
                current_scaler = scaler
                last_fit_idx = i
            except Exception as e:
                logger.warning(f"HMM fit failed at index {i}: {e}")
                continue

        if current_model is None or current_scaler is None:
            continue

        # Build full sequence up to t-1 to get proper Viterbi/forward path
        seq_start = max(0, i - 1 - config.window_days + 1)
        seq_end = i  # exclusive — so last element is i-1
        seq_data = clean.iloc[seq_start:seq_end][feature_cols].values
        seq_scaled = current_scaler.transform(seq_data)

        try:
            probs = current_model.predict_proba(seq_scaled)
            # Take the LAST row — state at t-1, aligned to prediction date t
            state_probs = probs[-1]

            results.append(HMMResult(
                trade_date=clean.index[i],
                state_probabilities=state_probs,
                most_likely_state=int(np.argmax(state_probs)),
                confidence=float(np.max(state_probs)),
                model_log_likelihood=float(current_model.score(seq_scaled)),
                window_start=clean.index[seq_start],
                window_end=clean.index[seq_end - 1],
            ))
        except Exception as e:
            logger.debug(f"HMM predict failed at {clean.index[i]}: {e}")

    if not results:
        logger.error("No HMM predictions were generated")
        return pd.DataFrame()

    rows = []
    for r in results:
        row = {"trade_date": r.trade_date}
        for s in range(config.n_states):
            row[f"state_{s}_prob"] = (
                r.state_probabilities[s] if s < len(r.state_probabilities) else 0.0
            )
        row["most_likely_state"] = r.most_likely_state
        row["confidence"] = r.confidence
        row["log_likelihood"] = r.model_log_likelihood
        rows.append(row)

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    logger.info(f"Generated {len(df)} predictions, avg confidence: {df['confidence'].mean():.3f}")
    return df


def fit_single_window(features: pd.DataFrame, config: HMMConfig):
    """Fit HMM on a single window (for analysis and state inspection).

    Returns (model, scaler, clean_data) tuple.
    """
    feature_cols = [c for c in features.columns if c in config.hmm_features]
    clean = features[feature_cols].dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean.values)

    model = hmm.GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
    model.fit(scaled)
    return model, scaler, clean
