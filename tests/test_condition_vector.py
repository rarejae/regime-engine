"""Tests for engine/condition_vector.py — ConditionVector builder."""

from __future__ import annotations

from datetime import datetime

import pytest

from data.models import AxisScores, IndicatorSnapshot, IndicatorValue
from engine.condition_vector import ConditionVector, build_condition_vector


def _make_axis_scores(
    growth: float = 0.0,
    inflation: float = 0.0,
    liquidity: float = 0.0,
    risk: float = 0.0,
) -> AxisScores:
    return AxisScores(
        GROWTH=growth, INFLATION=inflation, LIQUIDITY=liquidity, RISK=risk,
    )


def _make_snapshot(**indicators: float) -> IndicatorSnapshot:
    """Create a snapshot with the given indicator_id=value pairs."""
    snap = IndicatorSnapshot()
    for ind_id, val in indicators.items():
        snap.indicators[ind_id] = IndicatorValue(
            id=ind_id, value=val, axis="RISK", polarity="risk_on",
            weight=1.0, source="test", as_of=datetime.utcnow(),
        )
    return snap


class TestConditionVectorBuilder:
    """Tests for build_condition_vector."""

    def test_basic_build(self):
        axis = _make_axis_scores(growth=50, inflation=-20, liquidity=10, risk=-40)
        snap = _make_snapshot(VIX=25.0, VIX_TERM=2.5, VVIX=110.0, SKEW=135.0)

        cv = build_condition_vector(axis, snap, regime_label="GOLDILOCKS", conviction=60)

        assert cv.growth == 50.0
        assert cv.inflation == -20.0
        assert cv.liquidity == 10.0
        assert cv.risk == -40.0
        assert cv.vix == 25.0
        assert cv.vix_term == 2.5
        assert cv.vvix == 110.0
        assert cv.skew == 135.0
        assert cv.regime_label == "GOLDILOCKS"
        assert cv.conviction == 60.0

    def test_defaults_for_missing_indicators(self):
        axis = _make_axis_scores()
        snap = IndicatorSnapshot()  # empty

        cv = build_condition_vector(axis, snap)

        assert cv.vix == 20.0  # default
        assert cv.vix_term == 0.0
        assert cv.vvix == 100.0
        assert cv.skew == 130.0
        assert cv.move_index == 100.0

    def test_modifier_flags_tight_liq(self):
        axis = _make_axis_scores(liquidity=-50, risk=-40)
        snap = _make_snapshot(VIX=30.0, VIX_TERM=-5.0, VVIX=125.0)

        cv = build_condition_vector(axis, snap)

        assert cv.tight_liq is True
        assert cv.loose_liq is False
        assert cv.high_vol is True
        assert cv.low_vol is False
        assert cv.vol_crush_imminent is True  # vix_term < -3 and vvix > 120

    def test_modifier_flags_loose_liq(self):
        axis = _make_axis_scores(liquidity=50, risk=40)
        snap = _make_snapshot(VIX=12.0, VIX_TERM=3.0, VVIX=85.0)

        cv = build_condition_vector(axis, snap)

        assert cv.loose_liq is True
        assert cv.tight_liq is False
        assert cv.low_vol is True
        assert cv.high_vol is False
        assert cv.vol_crush_imminent is False

    def test_transition_deltas(self):
        axis_prev = _make_axis_scores(growth=20, inflation=10)
        axis_curr = _make_axis_scores(growth=40, inflation=-5)
        snap = _make_snapshot()

        cv = build_condition_vector(axis_curr, snap, prev_axis_scores=axis_prev)

        assert cv.growth_delta == 20.0  # 40 - 20
        assert cv.inflation_delta == -15.0  # -5 - 10

    def test_no_prev_scores_zero_deltas(self):
        axis = _make_axis_scores(growth=30)
        snap = _make_snapshot()

        cv = build_condition_vector(axis, snap, prev_axis_scores=None)

        assert cv.growth_delta == 0.0
        assert cv.inflation_delta == 0.0
        assert cv.liquidity_delta == 0.0
        assert cv.risk_delta == 0.0

    def test_credit_conditions_from_snapshot(self):
        axis = _make_axis_scores()
        snap = _make_snapshot(HY_OAS=500.0, IG_OAS=150.0, HY_IG_DIFF=350.0)

        cv = build_condition_vector(axis, snap)

        assert cv.hy_oas == 500.0
        assert cv.ig_oas == 150.0
        assert cv.hy_ig_diff == 350.0


class TestConditionVectorDataclass:
    """Tests for ConditionVector properties."""

    def test_all_fields_present(self):
        cv = ConditionVector(
            growth=50, inflation=20, liquidity=-10, risk=-30,
            growth_delta=5, inflation_delta=-3, liquidity_delta=0, risk_delta=-10,
            vix=28, vix_term=-2, vvix=115, skew=140, move_index=120,
            hy_oas=500, ig_oas=150, hy_ig_diff=350,
            tight_liq=False, loose_liq=False, high_vol=True, low_vol=False,
            vol_crush_imminent=False,
            regime_label="STAGFLATION", conviction=70,
        )

        assert cv.growth == 50
        assert cv.high_vol is True
        assert cv.regime_label == "STAGFLATION"
