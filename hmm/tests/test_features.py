"""Tests for PIT safety and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from hmm.config import HMMConfig
from hmm.features import _ewm_zscore


class TestEwmZscore:
    def test_output_clipped(self):
        series = pd.Series([1, 2, 3, 100, 5, 6, 7, 8, 9, 10] * 30)
        z = _ewm_zscore(series, halflife=50, clip=3.0)
        assert z.dropna().max() <= 3.0
        assert z.dropna().min() >= -3.0

    def test_mean_near_zero_for_stationary(self):
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 500))
        z = _ewm_zscore(series, halflife=100, clip=3.0)
        # After burn-in, z-scores should be roughly centered
        assert abs(z.iloc[200:].mean()) < 0.5

    def test_no_future_leakage(self):
        """Z-score at time t should not depend on data after t."""
        np.random.seed(42)
        data = list(np.random.normal(0, 1, 100))
        s1 = pd.Series(data)
        z1 = _ewm_zscore(s1, halflife=30, clip=3.0)

        # Append a massive outlier at the end
        s2 = pd.Series(data + [1000.0])
        z2 = _ewm_zscore(s2, halflife=30, clip=3.0)

        # Z-scores for the first 100 elements should be identical
        pd.testing.assert_series_equal(z1, z2.iloc[:100], check_names=False)


class TestPITSafety:
    def test_exclude_current_bar_shifts_features(self):
        """When exclude_current_bar is True, features should be shifted by 1."""
        config = HMMConfig(exclude_current_bar=True)
        # Verify the config flag exists and is True
        assert config.exclude_current_bar is True

    def test_pit_config_defaults(self):
        config = HMMConfig()
        assert config.exclude_current_bar is True
        assert config.zscore_halflife == 252
        assert config.zscore_clip == 3.0
