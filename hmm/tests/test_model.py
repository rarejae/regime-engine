"""Tests for v3 HMM model."""

import numpy as np
import pandas as pd
import pytest
from hmm.config import HMMConfig
from hmm.model import fit_and_predict_rolling

FEAT_NAMES = ["nfci_z", "wti_z", "ust_10y3m_z", "mort_30y_z",
              "fed_bs_z", "spy_rtn_21d", "vvix_z", "skew_z"]


@pytest.fixture
def synthetic_features():
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=1500, freq="D")
    regime = np.zeros(1500)
    regime[400:500] = 1
    regime[900:1000] = 1
    regime[1100:1200] = 2

    f = pd.DataFrame(index=dates)
    for name in FEAT_NAMES:
        base = np.random.normal(0, 0.5, 1500)
        if name in ("nfci_z", "vvix_z"):
            base = np.where(regime == 1, base + 1.5, base)
        elif name == "spy_rtn_21d":
            base = np.where(regime == 1, np.random.normal(-0.05, 0.04, 1500), np.random.normal(0.02, 0.03, 1500))
        f[name] = base
    return f, regime


def test_prediction_shape(synthetic_features):
    f, _ = synthetic_features
    cfg = HMMConfig(n_states=3, window_days=400, min_window_days=300, refit_frequency=50)
    p, _ = fit_and_predict_rolling(f, cfg)
    assert "trade_date" in p.columns
    for s in range(3):
        assert f"state_{s}_prob" in p.columns
    assert len(p) > 0


def test_probs_sum_to_one(synthetic_features):
    f, _ = synthetic_features
    cfg = HMMConfig(n_states=3, window_days=400, min_window_days=300, refit_frequency=50)
    p, _ = fit_and_predict_rolling(f, cfg)
    cols = [c for c in p.columns if c.endswith("_prob")]
    assert np.allclose(p[cols].sum(axis=1), 1.0, atol=1e-5)


def test_probs_bounded(synthetic_features):
    f, _ = synthetic_features
    cfg = HMMConfig(n_states=3, window_days=400, min_window_days=300, refit_frequency=50)
    p, _ = fit_and_predict_rolling(f, cfg)
    cols = [c for c in p.columns if c.endswith("_prob")]
    assert (p[cols] >= 0).all().all()
    assert (p[cols] <= 1).all().all()


def test_states_used(synthetic_features):
    f, _ = synthetic_features
    cfg = HMMConfig(n_states=3, window_days=400, min_window_days=300, refit_frequency=50)
    p, _ = fit_and_predict_rolling(f, cfg)
    assert p["most_likely_state"].nunique() >= 2


def test_stress_detection(synthetic_features):
    f, _ = synthetic_features
    cfg = HMMConfig(n_states=3, window_days=400, min_window_days=300, refit_frequency=21)
    p, _ = fit_and_predict_rolling(f, cfg)
    p = p.set_index("trade_date")
    stress = p.reindex(f.index[420:480]).dropna()
    calm = p.reindex(f.index[200:300]).dropna()
    if len(stress) > 0 and len(calm) > 0:
        assert stress["most_likely_state"].mode()[0] != calm["most_likely_state"].mode()[0]
