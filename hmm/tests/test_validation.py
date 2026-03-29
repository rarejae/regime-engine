"""Tests for walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from hmm.config import HMMConfig
from hmm.validation import walk_forward_validate, _count_params, _state_run_lengths


class TestCountParams:
    def test_full_covariance(self):
        # 2 states, 3 features, full covariance
        n = _count_params(2, 3, "full")
        # means: 2*3=6, trans: 2*1=2, start: 1, cov: 2*3*4/2=12
        assert n == 6 + 2 + 1 + 12

    def test_diag_covariance(self):
        n = _count_params(2, 3, "diag")
        assert n == 6 + 2 + 1 + 6  # cov: 2*3=6


class TestStateRunLengths:
    def test_simple(self):
        states = np.array([0, 0, 1, 1, 1, 0])
        assert _state_run_lengths(states) == [2, 3, 1]

    def test_single(self):
        states = np.array([0])
        assert _state_run_lengths(states) == [1]

    def test_empty(self):
        assert _state_run_lengths(np.array([])) == []


class TestWalkForwardValidate:
    def test_produces_folds(self):
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", periods=2000, freq="D")
        features = pd.DataFrame(index=dates)
        features["spy_return_21d"] = np.random.normal(0.01, 0.05, 2000)
        features["spy_vol_21d"] = np.random.normal(0.15, 0.03, 2000)
        features["vix_z"] = np.random.normal(0, 1, 2000)
        features["yield_curve_z"] = np.random.normal(0, 1, 2000)
        features["credit_spread_z"] = np.random.normal(0, 1, 2000)
        features["inflation_z"] = np.random.normal(0, 1, 2000)

        spy_prices = pd.Series(
            np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 2000))),
            index=dates,
        ) * 300

        config = HMMConfig(
            n_states=2, window_days=500, min_window_days=400,
            walk_forward_test_days=63, walk_forward_step_days=63,
            min_train_days=500,
        )
        report = walk_forward_validate(features, spy_prices, config)

        assert report.n_folds > 0
        assert report.avg_bic != 0
        assert report.avg_state_duration > 0
