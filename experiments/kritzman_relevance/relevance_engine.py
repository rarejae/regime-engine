"""Core relevance computation per Kritzman, Kulasekaran & Turkington (2023).

Relevance = Similarity + Informativeness, using Mahalanobis distance.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def compute_indicator_covariance(
    indicators: pd.DataFrame, as_of_date: pd.Timestamp, reg_eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute expanding-window covariance matrix and its inverse.

    Uses Ledoit-Wolf shrinkage for stability. Falls back to ridge
    regularization if shrinkage fails.

    Args:
        indicators: DataFrame of indicator z-scores, DatetimeIndex.
        as_of_date: Use data strictly before this date.
        reg_eps: Ridge regularization epsilon.

    Returns:
        (omega, omega_inv, mean_vec) — covariance matrix, its inverse,
        and the expanding-window mean vector.
    """
    data = indicators[indicators.index < as_of_date].dropna()
    assert len(data) > 0, f"No data before {as_of_date}"

    n, p = data.shape
    mean_vec = data.values.mean(axis=0)

    if n < p + 5:
        # Too few observations for reliable covariance — use identity
        logger.warning(f"Only {n} obs for {p} indicators at {as_of_date.date()}, using identity")
        omega = np.eye(p)
        omega_inv = np.eye(p)
        return omega, omega_inv, mean_vec

    # Ledoit-Wolf shrinkage
    try:
        lw = LedoitWolf().fit(data.values)
        omega = lw.covariance_
    except Exception:
        omega = np.cov(data.values, rowvar=False)

    # Ridge regularization for invertibility
    omega += np.eye(p) * reg_eps

    cond = np.linalg.cond(omega)
    if cond > 1e8:
        logger.warning(f"High condition number {cond:.0e} at {as_of_date.date()}")

    try:
        omega_inv = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        omega_inv = np.linalg.pinv(omega)

    return omega, omega_inv, mean_vec


def _mahalanobis_sq(a: np.ndarray, b: np.ndarray, omega_inv: np.ndarray) -> float:
    """Squared Mahalanobis distance between vectors a and b."""
    diff = a - b
    return float(diff @ omega_inv @ diff)


def compute_relevance(
    current: np.ndarray,
    candidates: pd.DataFrame,
    omega_inv: np.ndarray,
    mean_vec: np.ndarray,
) -> pd.Series:
    """Compute relevance of every candidate month relative to the current state.

    relevance(i, t) = similarity(x_i, x_t) + 0.5 * (info(x_i) + info(x_t))

    Where:
        similarity = -0.5 * (x_i - x_t)' Ω⁻¹ (x_i - x_t)
        info(x)    = (x - x̄)' Ω⁻¹ (x - x̄)

    Args:
        current: Current month's indicator vector (p,).
        candidates: DataFrame of historical indicator z-scores.
        omega_inv: Inverse covariance matrix (p, p).
        mean_vec: Expanding-window mean of indicators (p,).

    Returns:
        Series of relevance scores indexed by candidate dates.
    """
    x_t = current.astype(np.float64)
    x_bar = mean_vec.astype(np.float64)
    Oi = omega_inv.astype(np.float64)
    cand_vals = candidates.values.astype(np.float64)

    # Informativeness of current state (constant across candidates)
    info_t = float((x_t - x_bar) @ Oi @ (x_t - x_bar))

    # Vectorized over candidates
    diffs_sim = cand_vals - x_t  # (N, p)
    diffs_info = cand_vals - x_bar  # (N, p)

    similarity = -0.5 * np.sum((diffs_sim @ Oi) * diffs_sim, axis=1)
    info_i = np.sum((diffs_info @ Oi) * diffs_info, axis=1)

    relevance = similarity + 0.5 * (info_i + info_t)

    return pd.Series(relevance, index=candidates.index, name="relevance")


def select_relevant_subsample(
    relevance: pd.Series, top_pct: float = 0.20
) -> tuple[pd.Index, pd.Series]:
    """Select the top_pct most relevant observations.

    Args:
        relevance: Series of relevance scores.
        top_pct: Fraction of observations to keep.

    Returns:
        (selected_dates, selected_relevance)
    """
    n_select = max(int(len(relevance) * top_pct), 5)
    threshold = relevance.nlargest(n_select).iloc[-1]
    mask = relevance >= threshold
    selected = relevance[mask].sort_values(ascending=False).head(n_select)
    return selected.index, selected


def compute_relevance_weights(
    full_relevance: pd.Series,
    subsample_relevance: pd.Series,
) -> pd.Series:
    """Compute normalized relevance weights with lambda^2 bias correction.

    The partial-sample regression formula weights observations by relevance
    with a variance-ratio correction factor:

        lambda^2 = var(relevance_full) / var(relevance_subsample)

    This compensates for the selection bias introduced by keeping only
    the most relevant observations.

    Args:
        full_relevance: Relevance scores for ALL candidate months.
        subsample_relevance: Relevance scores for selected months only.

    Returns:
        Normalized weights that sum to 1.
    """
    var_full = full_relevance.var()
    var_sub = subsample_relevance.var()

    # Lambda^2 correction
    if var_sub > 0 and var_full > 0:
        lambda_sq = var_full / var_sub
    else:
        lambda_sq = 1.0

    # Weight = lambda^2 * relevance (shifted so min is positive)
    shifted = subsample_relevance - subsample_relevance.min() + 1e-10
    raw_weights = lambda_sq * shifted

    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()

    return weights
