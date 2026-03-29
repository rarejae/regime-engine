"""Tests for v3.1 state labeling with relative direction classification."""

import pandas as pd
import pytest


def test_days_in_state():
    from hmm.labeler import _compute_days_in_state
    s = pd.Series([0, 0, 0, 1, 1, 0, 0])
    assert list(_compute_days_in_state(s)) == [1, 2, 3, 1, 2, 1, 2]


def test_classify_excess():
    from hmm.labeler import _classify_excess
    # unconditional mean = 0.007 (typical monthly SPY return)
    assert _classify_excess(0.015, 0.007) == "bullish"   # excess = +0.008 > threshold
    assert _classify_excess(0.002, 0.007) == "bearish"   # excess = -0.005 < -threshold
    assert _classify_excess(0.008, 0.007) == "neutral"   # excess = +0.001 < threshold
    assert _classify_excess(float("nan"), 0.007) == "neutral"


def test_runs():
    from hmm.labeler import _compute_runs
    s = pd.Series([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    assert _compute_runs(s, 0) == [3, 4]
    assert _compute_runs(s, 1) == [2, 1]


def test_direction_accuracy_excess():
    from hmm.labeler import _direction_accuracy_excess
    import pandas as pd
    # Returns: 0.01, 0.02, -0.01, 0.015 with unconditional mean 0.007
    # Excess: +0.003, +0.013, -0.017, +0.008
    # For bullish: 3 of 4 have positive excess = 75%
    returns = pd.Series([0.01, 0.02, -0.01, 0.015])
    acc = _direction_accuracy_excess(returns, "bullish", 0.007)
    assert acc == 0.75

    # For bearish: 1 of 4 has negative excess = 25%
    acc_b = _direction_accuracy_excess(returns, "bearish", 0.007)
    assert acc_b == 0.25
