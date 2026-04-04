"""Regime-conditioned return and covariance estimates.

Uses relevance weights to compute regime-specific expected returns,
volatilities, and the full covariance matrix for the asset universe.
"""

import numpy as np
import pandas as pd


def conditioned_expected_returns(
    asset_returns: pd.DataFrame,
    relevant_dates: pd.Index,
    weights: pd.Series,
) -> pd.Series:
    """Relevance-weighted mean forward return for each asset.

    This is the direct analog of Harvey's average forward return from
    similar months, but with relevance weighting instead of equal weighting.

    Args:
        asset_returns: DataFrame of monthly returns, indexed by date.
            These should be FORWARD returns (month after the indicator date).
        relevant_dates: Dates of relevant historical months.
        weights: Normalized relevance weights for those dates.

    Returns:
        Series of expected returns per asset.
    """
    # Align: only use dates where both returns and weights exist
    available = relevant_dates.intersection(asset_returns.index)
    if len(available) == 0:
        return pd.Series(0.0, index=asset_returns.columns)

    returns_sub = asset_returns.loc[available]
    weights_aligned = weights.reindex(available).dropna()
    common = returns_sub.index.intersection(weights_aligned.index)
    returns_sub = returns_sub.loc[common]
    weights_aligned = weights_aligned.loc[common]

    # Re-normalize weights after alignment
    w = weights_aligned.values
    if w.sum() > 0:
        w = w / w.sum()

    # Weighted mean return
    expected = pd.Series(0.0, index=asset_returns.columns)
    for col in asset_returns.columns:
        vals = returns_sub[col].values
        mask = ~np.isnan(vals)
        if mask.sum() > 0:
            w_valid = w[mask]
            if w_valid.sum() > 0:
                w_valid = w_valid / w_valid.sum()
                expected[col] = float(np.sum(w_valid * vals[mask]))

    return expected


def conditioned_covariance_matrix(
    asset_returns: pd.DataFrame,
    relevant_dates: pd.Index,
    weights: pd.Series,
    conditioned_means: pd.Series | None = None,
) -> pd.DataFrame:
    """Relevance-weighted covariance matrix.

    For each asset pair (i, j), compute:
        cov(i, j) = sum_k w_k * (r_ki - mu_i) * (r_kj - mu_j)

    where w_k are relevance weights and mu are conditioned means.

    Args:
        asset_returns: Monthly returns DataFrame.
        relevant_dates: Dates of relevant months.
        weights: Normalized relevance weights.
        conditioned_means: Pre-computed conditioned means.
            If None, computed from the same data.

    Returns:
        DataFrame covariance matrix (assets × assets).
    """
    available = relevant_dates.intersection(asset_returns.index)
    if len(available) < 3:
        # Too few observations — return identity-like matrix
        n = len(asset_returns.columns)
        return pd.DataFrame(
            np.eye(n) * 0.01, index=asset_returns.columns, columns=asset_returns.columns
        )

    returns_sub = asset_returns.loc[available].copy()
    weights_aligned = weights.reindex(available).dropna()
    common = returns_sub.index.intersection(weights_aligned.index)
    returns_sub = returns_sub.loc[common]
    weights_aligned = weights_aligned.loc[common]

    w = weights_aligned.values
    if w.sum() > 0:
        w = w / w.sum()

    # Fill NaN with 0 for covariance computation
    R = returns_sub.fillna(0).values  # (N, assets)
    assets = asset_returns.columns

    if conditioned_means is None:
        conditioned_means = pd.Series(
            np.sum(w[:, None] * R, axis=0), index=assets
        )

    mu = conditioned_means.reindex(assets).fillna(0).values  # (assets,)
    demean = R - mu  # (N, assets)

    # Weighted covariance: (demean.T @ diag(w) @ demean)
    cov = (demean * w[:, None]).T @ demean  # (assets, assets)

    cov_df = pd.DataFrame(cov, index=assets, columns=assets)

    # Ensure positive semi-definiteness
    cov_df = _nearest_pd(cov_df)

    return cov_df


def _nearest_pd(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure covariance matrix is positive semi-definite.

    Uses eigenvalue floor: set any negative eigenvalues to a small positive.
    """
    vals = cov_df.values
    eigvals, eigvecs = np.linalg.eigh(vals)

    if np.any(eigvals < 0):
        floor = max(1e-8, eigvals[eigvals > 0].min() * 0.01) if np.any(eigvals > 0) else 1e-8
        eigvals = np.maximum(eigvals, floor)
        vals = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Symmetrize
        vals = (vals + vals.T) / 2

    return pd.DataFrame(vals, index=cov_df.index, columns=cov_df.columns)


def conditioned_volatilities(cov_matrix: pd.DataFrame) -> pd.Series:
    """Extract annualized volatilities from covariance matrix."""
    monthly_var = np.diag(cov_matrix.values)
    monthly_vol = np.sqrt(np.maximum(monthly_var, 0))
    return pd.Series(monthly_vol * np.sqrt(12), index=cov_matrix.columns, name="ann_vol")


def conditioned_correlations(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix."""
    vol = np.sqrt(np.maximum(np.diag(cov_matrix.values), 1e-12))
    outer = np.outer(vol, vol)
    corr = cov_matrix.values / outer
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.columns)
