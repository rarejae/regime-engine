"""Tests for v3 walk-forward validation."""

import numpy as np
import pandas as pd
import pytest
from hmm.config import HMMConfig

FEAT_NAMES = ["nfci_z", "wti_z", "ust_10y3m_z", "mort_30y_z",
              "fed_bs_z", "spy_rtn_21d", "vvix_z", "skew_z"]


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=2000, freq="D")
    f = pd.DataFrame(index=dates)
    for col in FEAT_NAMES:
        f[col] = np.random.normal(0, 1, 2000).clip(-3, 3)
    spy = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 2000))), index=dates)
    return f, spy


def test_produces_folds(synthetic_data):
    f, spy = synthetic_data
    cfg = HMMConfig(n_states=3, window_days=500, min_window_days=400,
                   walk_forward_test_days=63, walk_forward_step_days=63, min_train_days=500)
    from hmm.validation import walk_forward_validate
    r = walk_forward_validate(f, spy, cfg)
    assert r.n_folds > 0
    assert r.avg_bic > 0


def test_bic_finite(synthetic_data):
    f, spy = synthetic_data
    cfg = HMMConfig(n_states=3, window_days=500, min_window_days=400,
                   walk_forward_test_days=63, walk_forward_step_days=63, min_train_days=500)
    from hmm.validation import walk_forward_validate
    r = walk_forward_validate(f, spy, cfg)
    assert np.isfinite(r.avg_bic)
