"""Tests for Euclidean distance similarity computation."""

import numpy as np
import pandas as pd
import pytest
from regime.config import RegimeConfig
from regime.similarity import compute_distances


@pytest.fixture
def synthetic_zdata():
    dates = pd.date_range("1975-01-01", periods=600, freq="MS")
    np.random.seed(42)
    return pd.DataFrame({
        "sp500_z": np.random.normal(0, 1, 600).clip(-3, 3),
        "yield_curve_z": np.random.normal(0, 1, 600).clip(-3, 3),
        "copper_z": np.random.normal(0, 1, 600).clip(-3, 3),
    }, index=dates)


def test_distance_nonnegative(synthetic_zdata):
    config = RegimeConfig()
    target = synthetic_zdata.index[-1]
    result = compute_distances(synthetic_zdata, target, config)
    assert (result.distances >= 0).all()


def test_excludes_trailing_36_months(synthetic_zdata):
    config = RegimeConfig()
    target = synthetic_zdata.index[-1]
    result = compute_distances(synthetic_zdata, target, config)

    cutoff = target - pd.DateOffset(months=config.exclude_trailing_months)
    assert all(d <= cutoff for d in result.distances.index)


def test_similar_subset_smaller(synthetic_zdata):
    config = RegimeConfig()
    target = synthetic_zdata.index[-1]
    result = compute_distances(synthetic_zdata, target, config)

    total_candidates = len(result.distances)
    assert len(result.similar_dates) <= total_candidates * config.similar_quantile + 2
    assert len(result.similar_dates) > 0


def test_closest_has_min_distance(synthetic_zdata):
    config = RegimeConfig()
    target = synthetic_zdata.index[-1]
    result = compute_distances(synthetic_zdata, target, config)

    assert result.closest_date == result.distances.idxmin()
    assert abs(result.min_distance - result.distances.min()) < 1e-10


def test_identical_point_has_zero_distance(synthetic_zdata):
    """If we duplicate a point, distance to itself should be 0."""
    config = RegimeConfig(exclude_trailing_months=0)  # Disable exclusion for this test
    # Add a duplicate row far in the past
    dup_date = synthetic_zdata.index[50]
    target_date = synthetic_zdata.index[-1]

    # Make target identical to dup
    modified = synthetic_zdata.copy()
    modified.iloc[-1] = modified.iloc[50]

    result = compute_distances(modified, target_date, config)
    assert result.min_distance < 0.01  # Should be ~0
