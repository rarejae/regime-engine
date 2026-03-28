"""Tests for engine/strategy_scorer.py — strategy scoring against ConditionVector."""

from __future__ import annotations

import pytest

from engine.condition_vector import ConditionVector
from engine.strategy_scorer import (
    STRATEGIES,
    rank_strategies,
    score_strategy,
    select_dte,
    select_short_delta,
    strategy_type,
)


def _make_cv(**overrides) -> ConditionVector:
    """Create a ConditionVector with sensible defaults and optional overrides."""
    defaults = dict(
        growth=0.0, inflation=0.0, liquidity=0.0, risk=0.0,
        growth_delta=0.0, inflation_delta=0.0, liquidity_delta=0.0, risk_delta=0.0,
        vix=20.0, vix_term=2.0, vvix=100.0, skew=130.0, move_index=100.0,
        hy_oas=400.0, ig_oas=100.0, hy_ig_diff=300.0,
        tight_liq=False, loose_liq=False, high_vol=False, low_vol=False,
        vol_crush_imminent=False,
        regime_label="MIXED", conviction=50.0,
    )
    defaults.update(overrides)
    return ConditionVector(**defaults)


class TestScoreStrategy:
    """Tests for individual strategy scoring functions."""

    def test_bear_call_loves_stagflation(self):
        cv = _make_cv(growth=-60, inflation=50, risk=-40, vix=30, vix_term=-3, tight_liq=True)
        score = score_strategy("bear_call_spread", cv)
        assert score > 20  # strong positive score

    def test_bull_put_loves_goldilocks(self):
        cv = _make_cv(growth=60, inflation=-20, risk=20, vix=18, loose_liq=True, liquidity=40)
        score = score_strategy("bull_put_spread", cv)
        assert score > 15

    def test_bull_call_wants_low_vol(self):
        cv_low_vol = _make_cv(growth=50, vix=14, low_vol=True, risk=35)
        cv_high_vol = _make_cv(growth=50, vix=35, high_vol=True, risk=-40)

        score_low = score_strategy("bull_call_spread", cv_low_vol)
        score_high = score_strategy("bull_call_spread", cv_high_vol)
        assert score_low > score_high  # debit spreads prefer low vol

    def test_iron_condor_penalized_by_strong_direction(self):
        cv_neutral = _make_cv(growth=5, inflation=5, growth_delta=0, inflation_delta=0, vix=22)
        cv_directional = _make_cv(growth=70, inflation=60, growth_delta=20, inflation_delta=15)

        score_neutral = score_strategy("iron_condor", cv_neutral)
        score_directional = score_strategy("iron_condor", cv_directional)
        assert score_neutral > score_directional

    def test_bull_put_penalized_in_crash(self):
        # The VIX > 35 penalty reduces vol_edge for bull_put_spread.
        # Compare VIX just above vs just below the 35 threshold
        # with same risk level to isolate the penalty effect.
        cv_above = _make_cv(growth=30, vix=36, risk=-30)
        cv_below = _make_cv(growth=30, vix=34, risk=-30)

        score_above = score_strategy("bull_put_spread", cv_above)
        score_below = score_strategy("bull_put_spread", cv_below)
        # Penalty of -(36-35)*0.8 = -0.8 kicks in above VIX 35
        assert score_below > score_above

    def test_bear_call_vol_edge_halved_when_cheap(self):
        cv_expensive = _make_cv(vix=25, vix_term=-2, vvix=110, growth=-30)
        cv_cheap = _make_cv(vix=25, vix_term=8, vvix=80, growth=-30)

        score_expensive = score_strategy("bear_call_spread", cv_expensive)
        score_cheap = score_strategy("bear_call_spread", cv_cheap)
        assert score_expensive > score_cheap  # cheap vol = halved vol edge


class TestRankStrategies:
    """Tests for strategy ranking."""

    def test_returns_5_strategies_sorted(self):
        cv = _make_cv()
        ranked = rank_strategies(cv)

        assert len(ranked) == 5
        # Check sorted descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_all_strategies_present(self):
        cv = _make_cv()
        ranked = rank_strategies(cv)
        strategies = {s for s, _ in ranked}
        assert strategies == set(STRATEGIES)

    def test_stagflation_favors_bear_call(self):
        cv = _make_cv(growth=-60, inflation=50, vix=30, tight_liq=True, risk=-30)
        ranked = rank_strategies(cv)
        assert ranked[0][0] == "bear_call_spread"

    def test_goldilocks_favors_bullish(self):
        cv = _make_cv(growth=60, inflation=-20, vix=15, loose_liq=True, liquidity=40, low_vol=True, risk=35)
        ranked = rank_strategies(cv)
        # Either bull_put or bull_call should be top
        assert ranked[0][0] in ("bull_put_spread", "bull_call_spread")


class TestSelectShortDelta:
    """Tests for delta selection from ConditionVector."""

    def test_high_conviction_higher_delta(self):
        cv_high = _make_cv(growth=80, inflation=50)
        cv_low = _make_cv(growth=15, inflation=5)

        delta_high = select_short_delta(cv_high, "bear_call_spread")
        delta_low = select_short_delta(cv_low, "bear_call_spread")
        assert delta_high > delta_low

    def test_delta_bounds(self):
        cv = _make_cv(growth=100, inflation=100)  # extreme
        delta = select_short_delta(cv, "bear_call_spread")
        assert 0.08 <= delta <= 0.40

    def test_skew_adjustment_for_puts(self):
        cv_high_skew = _make_cv(growth=40, skew=145)
        cv_low_skew = _make_cv(growth=40, skew=115)

        delta_high = select_short_delta(cv_high_skew, "bull_put_spread")
        delta_low = select_short_delta(cv_low_skew, "bull_put_spread")
        assert delta_low > delta_high  # high skew → lower delta

    def test_momentum_adjustment(self):
        cv_momentum = _make_cv(growth=40, growth_delta=10)
        cv_flat = _make_cv(growth=40, growth_delta=0)

        delta_m = select_short_delta(cv_momentum, "bull_put_spread")
        delta_f = select_short_delta(cv_flat, "bull_put_spread")
        assert delta_m > delta_f  # positive momentum → more aggressive


class TestSelectDTE:
    """Tests for DTE selection from ConditionVector."""

    def test_credit_shorter_than_debit(self):
        cv = _make_cv()
        dte_credit = select_dte(cv, "credit")
        dte_debit = select_dte(cv, "debit")
        assert dte_credit < dte_debit

    def test_tight_liq_shorter_dte(self):
        cv_tight = _make_cv(tight_liq=True)
        cv_normal = _make_cv()

        dte_tight = select_dte(cv_tight, "credit")
        dte_normal = select_dte(cv_normal, "credit")
        assert dte_tight < dte_normal

    def test_dte_bounds(self):
        cv = _make_cv(tight_liq=True, vix_term=-5, growth_delta=50, inflation_delta=50)
        dte = select_dte(cv, "credit")
        assert 14 <= dte <= 75

    def test_backwardation_shorter_dte(self):
        cv_back = _make_cv(vix_term=-5)
        cv_contango = _make_cv(vix_term=6)

        dte_back = select_dte(cv_back, "credit")
        dte_contango = select_dte(cv_contango, "credit")
        assert dte_back < dte_contango


class TestStrategyType:
    """Tests for strategy metadata."""

    def test_credit_strategies(self):
        assert strategy_type("bull_put_spread") == "credit"
        assert strategy_type("bear_call_spread") == "credit"
        assert strategy_type("iron_condor") == "credit"

    def test_debit_strategies(self):
        assert strategy_type("bull_call_spread") == "debit"
        assert strategy_type("bear_put_spread") == "debit"
