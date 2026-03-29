"""Tests for v3 feature engineering."""

import numpy as np
import pandas as pd
import pytest
from hmm.config import HMMConfig


@pytest.fixture
def synthetic_raw(tmp_path):
    dates = pd.date_range("2010-01-01", periods=3000, freq="D")
    np.random.seed(42)
    raw = pd.DataFrame(index=dates)
    raw["NFCI"] = np.random.normal(0, 0.5, 3000)
    raw["WTI"] = np.random.lognormal(3.5, 0.3, 3000)
    raw["UST_10Y_3M"] = np.random.normal(1.5, 1.0, 3000)
    raw["MORT_30Y"] = np.random.normal(4.5, 1.0, 3000)
    raw["FED_BS_DELTA_3M"] = np.cumsum(np.random.normal(0, 0.5, 3000))
    raw["SPY_CLOSE"] = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 3000)))
    raw["VVIX"] = np.random.lognormal(4.5, 0.2, 3000)
    raw["SKEW"] = np.random.normal(130, 10, 3000)
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path)
    return path


def test_all_features_present(synthetic_raw):
    config = HMMConfig(raw_indicators_path=synthetic_raw)
    from hmm.features import build_hmm_features
    features = build_hmm_features(config)
    for f in config.hmm_features:
        assert f in features.columns, f"Missing: {f}"


def test_pit_safety(synthetic_raw, tmp_path):
    config = HMMConfig(raw_indicators_path=synthetic_raw, exclude_current_bar=True)
    from hmm.features import build_hmm_features
    f1 = build_hmm_features(config)

    raw2 = pd.read_parquet(synthetic_raw)
    raw2.iloc[-1, raw2.columns.get_loc("NFCI")] = 10.0
    p2 = tmp_path / "raw2.parquet"
    raw2.to_parquet(p2)
    config2 = HMMConfig(raw_indicators_path=p2, exclude_current_bar=True)
    f2 = build_hmm_features(config2)

    last = f1.index[-1]
    if last in f2.index:
        assert f1.loc[last, "nfci_z"] == f2.loc[last, "nfci_z"], "PIT violation"


def test_zscore_bounded(synthetic_raw):
    config = HMMConfig(raw_indicators_path=synthetic_raw)
    from hmm.features import build_hmm_features
    features = build_hmm_features(config)
    for col in config.hmm_features:
        if col == "spy_rtn_21d":
            continue  # returns are not z-scored
        v = features[col].dropna()
        if len(v) > 0:
            assert v.max() <= 3.0 + 1e-10
            assert v.min() >= -3.0 - 1e-10


def test_high_coverage(synthetic_raw):
    config = HMMConfig(raw_indicators_path=synthetic_raw)
    from hmm.features import build_hmm_features
    features = build_hmm_features(config)
    for col in config.hmm_features:
        assert features[col].notna().mean() >= 0.80
