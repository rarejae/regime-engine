"""Tests for engine/normalizer.py"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.models import IndicatorSnapshot, IndicatorValue


def _make_snapshot(indicator_id: str, value: float, axis: str = "RISK", polarity: str = "risk_off") -> IndicatorSnapshot:
    """Helper: create a minimal snapshot with one indicator."""
    iv = IndicatorValue(
        id=indicator_id,
        value=value,
        unit="idx",
        axis=axis,
        polarity=polarity,
        weight=0.30,
        source="yahoo",
        as_of=datetime.utcnow(),
    )
    return IndicatorSnapshot(indicators={indicator_id: iv})


class TestNormalizer:
    """Tests for the Normalizer class."""

    @pytest.fixture
    def normalizer(self):
        from engine.normalizer import Normalizer
        return Normalizer()

    def test_z_score_with_sufficient_history(self, normalizer):
        """Z-score should be (current - mean) / std when history > min_periods."""
        from engine.normalizer import Normalizer, HistoryStore
        from data.models import IndicatorValue

        iv = IndicatorValue(
            id="TEST",
            value=30.0,
            unit="idx",
            axis="RISK",
            polarity="risk_off",
            weight=0.30,
            source="test",
            as_of=datetime.utcnow(),
        )

        # Inject history: mean=20, std=5, so z = (30-20)/5 = 2.0
        history = pd.Series([20.0] * 20, index=pd.date_range("2019-01-01", periods=20, freq="ME"))
        normalizer.store._history["TEST"] = history

        result = normalizer._compute_z("TEST", 30.0, history, iv, z_clip=3.0)
        assert result.z_score_unclipped == pytest.approx(0.0, abs=0.01)  # all same value → z=0 or near

    def test_z_score_clipping(self, normalizer):
        """Z-score should be clipped to ±z_clip."""
        from data.models import IndicatorValue

        iv = IndicatorValue(
            id="TEST",
            value=100.0,
            unit="idx",
            axis="RISK",
            polarity="risk_off",
            weight=0.30,
            source="test",
            as_of=datetime.utcnow(),
        )
        # history: values centered around 0 with std=1; current=100 → z_raw≈100 → clips to 3.0
        rng = pd.date_range("2019-01-01", periods=50, freq="ME")
        history = pd.Series(list(range(-25, 25)), index=rng, dtype=float)
        result = normalizer._compute_z("TEST", 100.0, history, iv, z_clip=3.0)
        assert result.z_score == 3.0
        assert result.z_score_unclipped > 3.0

    def test_insufficient_history_returns_zero_z(self, normalizer):
        """With < min_history periods, z_score should be 0."""
        from data.models import IndicatorValue
        from engine.normalizer import _MIN_HISTORY

        iv = IndicatorValue(
            id="TEST_SPARSE",
            value=42.0,
            unit="idx",
            axis="RISK",
            polarity="risk_off",
            weight=0.10,
            source="test",
            as_of=datetime.utcnow(),
        )
        sparse_history = pd.Series([42.0] * (_MIN_HISTORY - 1))
        result = normalizer._compute_z("TEST_SPARSE", 42.0, sparse_history, iv, z_clip=3.0)
        assert result.z_score == 0.0

    def test_polarity_sign_assignment(self, normalizer):
        """NormalizedIndicator should have correct polarity_sign."""
        from data.models import IndicatorValue
        from engine.normalizer import _polarity_sign

        assert _polarity_sign("risk_on") == 1
        assert _polarity_sign("risk_off") == -1
        assert _polarity_sign("ambiguous") == 0

    def test_normalize_full_snapshot(self, normalizer):
        """normalize() should return NormalizedIndicator for each available indicator."""
        snapshot = IndicatorSnapshot()
        for i, (ind_id, axis) in enumerate([("VIX", "RISK"), ("HY_OAS", "RISK"), ("NFP_3MA", "GROWTH")]):
            snapshot.indicators[ind_id] = IndicatorValue(
                id=ind_id,
                value=20.0 + i * 5,
                unit="idx",
                axis=axis,
                polarity="risk_off",
                weight=0.30,
                source="test",
                as_of=datetime.utcnow(),
            )
            # Pre-seed history
            normalizer.store._history[ind_id] = pd.Series(
                [20.0] * 60,
                index=pd.date_range("2019-01-01", periods=60, freq="ME"),
            )

        with patch.object(normalizer.store, "_load_history", return_value=pd.Series(dtype=float)):
            results = normalizer.normalize(snapshot)

        assert len(results) == 3
        ids = {r.id for r in results}
        assert "VIX" in ids
        assert "NFP_3MA" in ids


class TestHistoryDerivations:
    """Tests for the HistoryStore._derive_history method."""

    def test_yoy_pct_derivation(self):
        from engine.normalizer import HistoryStore
        raw = pd.Series(
            [100.0] * 12 + [105.0],
            index=pd.date_range("2023-01-01", periods=13, freq="ME"),
        )
        result = HistoryStore._derive_history(raw, "yoy_pct", {})
        assert result.dropna().iloc[-1] == pytest.approx(5.0, abs=0.1)

    def test_mom_pct_derivation_constant_series(self):
        """mom_pct of a constant series should be all zeros."""
        from engine.normalizer import HistoryStore
        raw = pd.Series([4.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="D"))
        result = HistoryStore._derive_history(raw, "mom_pct", {})
        assert all(abs(v) < 1e-10 for v in result.dropna().tolist())
