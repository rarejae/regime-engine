"""Euclidean distance similarity engine per Harvey-Mulliner (2025)."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime.config import RegimeConfig

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Similarity assessment for a single target month."""
    target_date: pd.Timestamp
    target_z: np.ndarray            # z-scores at target month
    distances: pd.Series            # distance to every candidate month
    similar_dates: list              # bottom quantile dates
    dissimilar_dates: list           # top quantile dates
    min_distance: float              # closest analog
    median_distance: float
    closest_date: pd.Timestamp
    closest_z: np.ndarray


def compute_distances(
    z_data: pd.DataFrame,
    target_date: pd.Timestamp,
    config: RegimeConfig,
) -> SimilarityResult:
    """Compute Euclidean distance from target month to all historical months.

    Excludes trailing 36 months (anti-momentum mask).
    """
    if target_date not in z_data.index:
        raise ValueError(f"{target_date} not in data")

    target_z = z_data.loc[target_date].values

    # Candidate pool: exclude trailing 36 months
    cutoff = target_date - pd.DateOffset(months=config.exclude_trailing_months)
    candidates = z_data[z_data.index <= cutoff]

    if len(candidates) == 0:
        raise ValueError(f"No candidates before {cutoff}")

    # Euclidean distance: d_Ti = sqrt(sum((x_iv - x_Tv)^2))
    diffs = candidates.values - target_z
    distances = pd.Series(
        np.sqrt(np.sum(diffs ** 2, axis=1)),
        index=candidates.index,
    )

    # Similar = bottom quantile, dissimilar = top quantile
    sim_thresh = distances.quantile(config.similar_quantile)
    dissim_thresh = distances.quantile(config.dissimilar_quantile)

    similar = distances[distances <= sim_thresh].sort_values()
    dissimilar = distances[distances >= dissim_thresh].sort_values(ascending=False)

    closest_idx = distances.idxmin()

    return SimilarityResult(
        target_date=target_date,
        target_z=target_z,
        distances=distances,
        similar_dates=similar.index.tolist(),
        dissimilar_dates=dissimilar.index.tolist(),
        min_distance=float(distances.min()),
        median_distance=float(distances.median()),
        closest_date=closest_idx,
        closest_z=z_data.loc[closest_idx].values,
    )


def compute_all_similarities(
    z_data: pd.DataFrame,
    config: RegimeConfig,
    start: str | None = None,
    end: str | None = None,
) -> list[SimilarityResult]:
    """Compute similarity for every month in the backtest range."""
    start_dt = pd.Timestamp(start or config.backtest_start)
    end_dt = pd.Timestamp(end or config.backtest_end)

    dates = z_data.index[(z_data.index >= start_dt) & (z_data.index <= end_dt)]
    results = []

    for dt in dates:
        try:
            r = compute_distances(z_data, dt, config)
            results.append(r)
        except ValueError as e:
            logger.debug(f"Skipping {dt}: {e}")

    logger.info(f"Computed similarity for {len(results)} months")
    return results
