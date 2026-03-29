"""Tests for Harvey-Mulliner variable transformation."""

import numpy as np
import pandas as pd
import pytest
from regime.config import RegimeConfig
from regime.transform import transform_variables, get_valid_zscored


@pytest.fixture
def synthetic_raw():
    dates = pd.date_range("1960-01-01", periods=800, freq="MS")
    np.random.seed(42)
    return pd.DataFrame({
        "sp500": np.cumsum(np.random.normal(0.5, 5, 800)) + 100,
        "yield_curve": np.random.normal(1.5, 1.0, 800),
        "copper": np.cumsum(np.random.normal(0, 0.5, 800)) + 100,
    }, index=dates)


def test_12month_change(synthetic_raw):
    config = RegimeConfig()
    result = transform_variables(synthetic_raw, config)
    # 12-month change should be current - 12 months ago
    for col in synthetic_raw.columns:
        chg_col = f"{col}_chg"
        assert chg_col in result.columns
        # Check a specific point (index 24 = 2 years in)
        expected = synthetic_raw[col].iloc[24] - synthetic_raw[col].iloc[12]
        actual = result[chg_col].iloc[24]
        assert abs(actual - expected) < 1e-10


def test_zscore_winsorized(synthetic_raw):
    config = RegimeConfig()
    result = transform_variables(synthetic_raw, config)
    for col in synthetic_raw.columns:
        z_col = f"{col}_z"
        valid = result[z_col].dropna()
        assert valid.max() <= 3.0 + 1e-10
        assert valid.min() >= -3.0 - 1e-10


def test_valid_zscored_drops_nans(synthetic_raw):
    config = RegimeConfig()
    result = transform_variables(synthetic_raw, config)
    valid = get_valid_zscored(result, config)
    assert valid.isna().sum().sum() == 0
    assert len(valid) < len(result)  # Should drop early rows
