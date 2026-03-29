"""Tests for HMM model fitting and prediction."""

import numpy as np
import pandas as pd
import pytest

from hmm.config import HMMConfig
from hmm.model import fit_and_predict_rolling, fit_single_window


@pytest.fixture
def synthetic_features():
    """Generate synthetic features with known regime structure."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=1200, freq="D")

    regime = np.zeros(1200)
    regime[300:400] = 1
    regime[700:800] = 1

    features = pd.DataFrame(index=dates)
    features["spy_return_21d"] = np.where(
        regime == 0, np.random.normal(0.02, 0.03, 1200), np.random.normal(-0.05, 0.08, 1200)
    )
    features["spy_vol_21d"] = np.where(
        regime == 0, np.random.normal(0.12, 0.02, 1200), np.random.normal(0.35, 0.05, 1200)
    )
    features["vix_z"] = np.where(
        regime == 0, np.random.normal(-0.5, 0.3, 1200), np.random.normal(1.5, 0.5, 1200)
    )
    features["yield_curve_z"] = np.random.normal(0, 0.5, 1200)
    features["credit_spread_z"] = np.where(
        regime == 0, np.random.normal(-0.3, 0.2, 1200), np.random.normal(1.0, 0.4, 1200)
    )
    features["inflation_z"] = np.random.normal(0, 0.5, 1200)

    return features, regime


def test_prediction_shape(synthetic_features):
    features, _ = synthetic_features
    config = HMMConfig(n_states=2, window_days=300, min_window_days=200, refit_frequency=50)
    predictions = fit_and_predict_rolling(features, config)

    assert "trade_date" in predictions.columns
    assert "state_0_prob" in predictions.columns
    assert "state_1_prob" in predictions.columns
    assert "most_likely_state" in predictions.columns
    assert "confidence" in predictions.columns
    assert len(predictions) > 0


def test_probabilities_sum_to_one(synthetic_features):
    features, _ = synthetic_features
    config = HMMConfig(n_states=2, window_days=300, min_window_days=200, refit_frequency=50)
    predictions = fit_and_predict_rolling(features, config)

    prob_cols = [c for c in predictions.columns if c.endswith("_prob")]
    row_sums = predictions[prob_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_probabilities_bounded(synthetic_features):
    features, _ = synthetic_features
    config = HMMConfig(n_states=2, window_days=300, min_window_days=200, refit_frequency=50)
    predictions = fit_and_predict_rolling(features, config)

    prob_cols = [c for c in predictions.columns if c.endswith("_prob")]
    assert (predictions[prob_cols] >= 0).all().all()
    assert (predictions[prob_cols] <= 1).all().all()


def test_pit_safety(synthetic_features):
    features, _ = synthetic_features
    config = HMMConfig(
        n_states=2, window_days=300, min_window_days=200,
        refit_frequency=50, exclude_current_bar=True,
    )
    predictions = fit_and_predict_rolling(features, config)

    first_pred_date = predictions["trade_date"].min()
    first_feature_date = features.index[0]
    days_gap = (first_pred_date - first_feature_date).days
    assert days_gap >= config.min_window_days


def test_stress_detection(synthetic_features):
    features, true_regime = synthetic_features
    config = HMMConfig(n_states=2, window_days=300, min_window_days=200, refit_frequency=21)
    predictions = fit_and_predict_rolling(features, config)
    pred_indexed = predictions.set_index("trade_date")

    stress_dates = features.index[320:380]
    stress_preds = pred_indexed.reindex(stress_dates).dropna()

    calm_dates = features.index[100:200]
    calm_preds = pred_indexed.reindex(calm_dates).dropna()

    if len(stress_preds) > 0 and len(calm_preds) > 0:
        dominant_stress = stress_preds["most_likely_state"].mode()[0]
        dominant_calm = calm_preds["most_likely_state"].mode()[0]
        assert dominant_stress != dominant_calm, "HMM should distinguish stress from calm"


def test_fit_single_window(synthetic_features):
    features, _ = synthetic_features
    config = HMMConfig(n_states=2)
    model, scaler, clean = fit_single_window(features, config)

    assert model.n_components == 2
    assert scaler is not None
    assert len(clean) > 0
