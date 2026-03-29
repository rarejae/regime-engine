"""Tests for state labeling consistency."""

import numpy as np
import pandas as pd
import pytest

from hmm.labeler import _compute_runs, _auto_label


class TestComputeRuns:
    def test_single_state(self):
        states = pd.Series([0, 0, 0, 0, 0])
        assert _compute_runs(states, 0) == [5]
        assert _compute_runs(states, 1) == []

    def test_alternating(self):
        states = pd.Series([0, 1, 0, 1, 0])
        assert _compute_runs(states, 0) == [1, 1, 1]
        assert _compute_runs(states, 1) == [1, 1]

    def test_long_run(self):
        states = pd.Series([1, 1, 1, 0, 0, 1, 1])
        assert _compute_runs(states, 1) == [3, 2]
        assert _compute_runs(states, 0) == [2]


class TestAutoLabel:
    def test_steady_bull(self):
        label = _auto_label(0.01, 0.03, {"vix_z": -0.5, "inflation_z": 0.0}, 0.3)
        assert label == "steady_bull"

    def test_stress_panic(self):
        label = _auto_label(-0.03, 0.08, {"vix_z": 2.0, "inflation_z": 0.0}, 0.1)
        assert label == "stress_panic"

    def test_recovery_snap(self):
        label = _auto_label(0.03, 0.06, {"vix_z": 1.0, "inflation_z": 0.0}, 0.1)
        assert label == "recovery_snap"

    def test_inflation_grind(self):
        label = _auto_label(0.003, 0.03, {"vix_z": 0.0, "inflation_z": 1.0}, 0.2)
        assert label == "inflation_grind"
