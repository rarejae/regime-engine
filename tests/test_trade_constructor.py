"""Tests for engine/trade_constructor.py, engine/strategy_selector.py,
engine/win_probability.py, and utils/helpers.py"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.models import AxisScores, RegimeClassification, TradeCard


# ── Helpers / Black-Scholes Tests ─────────────────────────────────────────────

class TestBlackScholes:
    """Tests for utils/helpers.bs_delta and bs_price."""

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be ~0.5."""
        from utils.helpers import bs_delta
        delta = bs_delta(S=500, K=500, T=30/365, r=0.05, sigma=0.18, option_type="call")
        assert 0.45 < delta < 0.55

    def test_atm_put_delta_near_neg_half(self):
        """ATM put delta should be ~-0.5."""
        from utils.helpers import bs_delta
        delta = bs_delta(S=500, K=500, T=30/365, r=0.05, sigma=0.18, option_type="put")
        assert -0.55 < delta < -0.45

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call delta should approach 1.0."""
        from utils.helpers import bs_delta
        delta = bs_delta(S=600, K=400, T=30/365, r=0.05, sigma=0.18, option_type="call")
        assert delta > 0.99

    def test_far_otm_put_delta_near_zero(self):
        """Far OTM put delta should approach 0."""
        from utils.helpers import bs_delta
        delta = bs_delta(S=500, K=300, T=21/365, r=0.05, sigma=0.18, option_type="put")
        assert abs(delta) < 0.01

    def test_put_call_parity_delta(self):
        """call_delta + |put_delta| ≈ 1 for same strike/expiry (delta parity)."""
        from utils.helpers import bs_delta
        call_d = bs_delta(500, 510, 21/365, 0.05, 0.18, "call")
        put_d = bs_delta(500, 510, 21/365, 0.05, 0.18, "put")
        # call_delta - abs(put_delta) = 1  ↔  call_delta + put_delta_signed = 1
        assert abs(call_d + abs(put_d) - 1.0) < 0.001

    def test_zero_vol_raises(self):
        """bs_delta should raise for sigma=0."""
        from utils.helpers import bs_delta
        with pytest.raises(ValueError):
            bs_delta(500, 500, 21/365, 0.05, 0.0, "call")

    def test_expired_call_delta(self):
        """T=0 call: delta=1 if ITM, 0 if OTM."""
        from utils.helpers import bs_delta
        assert bs_delta(510, 500, 0, 0.05, 0.18, "call") == 1.0
        assert bs_delta(490, 500, 0, 0.05, 0.18, "call") == 0.0

    def test_bs_price_positive(self):
        """bs_price should return positive values for reasonable inputs."""
        from utils.helpers import bs_price
        price = bs_price(500, 510, 21/365, 0.05, 0.18, "call")
        assert price > 0

    def test_bs_price_put_call_parity(self):
        """P-C parity: C - P = S - K*e^(-rT)."""
        from utils.helpers import bs_price
        import math
        S, K, T, r, sigma = 500.0, 500.0, 30/365, 0.05, 0.18
        C = bs_price(S, K, T, r, sigma, "call")
        P = bs_price(S, K, T, r, sigma, "put")
        pcp = C - P
        theoretical = S - K * math.exp(-r * T)
        assert abs(pcp - theoretical) < 0.01


class TestHelpers:
    """Tests for utils/helpers — date, chain, and option utilities."""

    def test_target_expiry_date_picks_nearest(self):
        """Should pick the expiry closest to target_dte."""
        from utils.helpers import target_expiry_date
        today = date.today()
        expiries = [
            (today + timedelta(days=14)).strftime("%Y-%m-%d"),
            (today + timedelta(days=21)).strftime("%Y-%m-%d"),
            (today + timedelta(days=35)).strftime("%Y-%m-%d"),
        ]
        result = target_expiry_date(21, expiries)
        assert result == expiries[1]

    def test_target_expiry_raises_on_empty(self):
        """Should raise ValueError if no expirations provided."""
        from utils.helpers import target_expiry_date
        with pytest.raises(ValueError):
            target_expiry_date(21, [])

    def test_dte_for_expiry_future(self):
        """DTE should be positive for future date."""
        from utils.helpers import dte_for_expiry
        future = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
        assert dte_for_expiry(future) == 30

    def test_years_to_expiry(self):
        """years_to_expiry should convert DTE to year fraction."""
        from utils.helpers import years_to_expiry
        future = (date.today() + timedelta(days=365)).strftime("%Y-%m-%d")
        assert abs(years_to_expiry(future) - 1.0) < 0.01

    def test_find_strike_by_delta_returns_closest(self):
        """find_strike_by_delta should return strike with delta closest to target."""
        from utils.helpers import find_strike_by_delta
        chain = pd.DataFrame({
            "strike": [480.0, 490.0, 500.0, 510.0],
            "bid": [0.50, 1.00, 2.00, 3.50],
            "ask": [0.55, 1.05, 2.05, 3.55],
            "mid": [0.525, 1.025, 2.025, 3.525],
            "impliedVolatility": [0.12, 0.15, 0.18, 0.22],
            "openInterest": [500, 800, 1200, 700],
            "volume": [100, 200, 300, 150],
        })
        # Target delta=0.30 for put (OTM)
        result = find_strike_by_delta(chain, 0.30, S=500, T=21/365, r=0.05, option_type="put")
        assert result is not None
        assert "strike" in result.index


# ── StrategySelector Tests ────────────────────────────────────────────────────

class TestStrategySelector:
    """Tests for engine/strategy_selector.py"""

    @pytest.fixture
    def selector(self):
        from engine.strategy_selector import StrategySelector
        return StrategySelector()

    def _make_regime(self, primary: str, conviction: float = 60.0) -> RegimeClassification:
        return RegimeClassification(
            primary=primary,
            modifiers=[],
            conviction=conviction,
            transition="stable",
            axis_scores=AxisScores(GROWTH=0, INFLATION=0, LIQUIDITY=0, RISK=0),
        )

    def test_goldilocks_bull_put_spread(self, selector):
        result = selector.select(self._make_regime("GOLDILOCKS"), vix=18.0)
        assert result["strategy"] == "Bull Put Spread"
        assert result["strategy_type"] == "credit"

    def test_reflation_bull_call_spread(self, selector):
        result = selector.select(self._make_regime("REFLATION"), vix=20.0)
        assert result["strategy"] == "Bull Call Spread"
        assert result["strategy_type"] == "debit"

    def test_stagflation_bear_call_spread(self, selector):
        result = selector.select(self._make_regime("STAGFLATION"), vix=25.0)
        assert result["strategy"] == "Bear Call Spread"
        assert result["strategy_type"] == "credit"

    def test_deflation_bear_put_spread(self, selector):
        result = selector.select(self._make_regime("DEFLATION"), vix=35.0)
        assert result["strategy"] == "Bear Put Spread"
        assert result["strategy_type"] == "debit"

    def test_mixed_iron_condor(self, selector):
        result = selector.select(self._make_regime("MIXED"), vix=15.0)
        assert result["strategy"] == "Iron Condor"
        assert result["strategy_type"] == "credit"

    def test_high_conviction_shorter_dte_credit(self, selector):
        """High conviction credit spread should use shorter DTE (21d)."""
        result = selector.select(self._make_regime("GOLDILOCKS", conviction=75), vix=15.0)
        assert result["dte"] == 21

    def test_low_conviction_longer_dte_credit(self, selector):
        """Low conviction credit spread should use longer DTE (45d)."""
        result = selector.select(self._make_regime("GOLDILOCKS", conviction=10), vix=15.0)
        assert result["dte"] == 45

    def test_width_range_scales_with_vix(self, selector):
        """Width range should increase with VIX level."""
        low_vix = selector.select(self._make_regime("GOLDILOCKS"), vix=12.0)["width_range"]
        high_vix = selector.select(self._make_regime("GOLDILOCKS"), vix=30.0)["width_range"]
        assert low_vix["max"] < high_vix["max"]


# ── WinProbabilityModel Tests ─────────────────────────────────────────────────

class TestWinProbabilityModel:
    """Tests for engine/win_probability.py"""

    @pytest.fixture
    def model(self):
        from engine.win_probability import WinProbabilityModel
        return WinProbabilityModel()

    def _make_card(self, strategy: str, strategy_type: str, short_delta: float = 0.25, long_delta: float = 0.10) -> TradeCard:
        return TradeCard(
            regime="GOLDILOCKS",
            modifiers=[],
            conviction=60.0,
            transition="stable",
            strategy=strategy,
            strategy_type=strategy_type,
            underlying="SPY",
            short_strike=490.0,
            short_delta=short_delta,
            long_strike=485.0,
            long_delta=long_delta,
            width=5.0,
            expiry="2025-04-18",
            dte=21,
            estimated_credit=1.50,
            max_risk=3.50,
            credit_to_width=0.30,
            breakeven=488.50,
            win_probability=0.0,
            axis_scores={"GROWTH": 0, "INFLATION": 0, "LIQUIDITY": 0, "RISK": 0},
            timestamp="2025-03-28T00:00:00",
        )

    def _make_regime(self, primary: str, modifiers: list[str] = None) -> RegimeClassification:
        return RegimeClassification(
            primary=primary,
            modifiers=modifiers or [],
            conviction=60.0,
            transition="stable",
            axis_scores=AxisScores(GROWTH=0, INFLATION=0, LIQUIDITY=0, RISK=0),
        )

    def test_credit_spread_base_probability(self, model):
        """Credit spread win prob base = 1 - |short_delta|."""
        card = self._make_card("Bull Put Spread", "credit", short_delta=0.25)
        regime = self._make_regime("GOLDILOCKS")
        result = model.compute(card, regime)
        # base = 0.75, aligned bonus = +0.05
        assert result.win_probability == pytest.approx(0.80, abs=0.01)

    def test_debit_spread_base_probability(self, model):
        """Debit spread win prob base = |long_delta|."""
        card = self._make_card("Bull Call Spread", "debit", long_delta=0.45)
        regime = self._make_regime("REFLATION")
        result = model.compute(card, regime)
        # base = 0.45, aligned bonus = +0.05
        assert result.win_probability == pytest.approx(0.50, abs=0.02)

    def test_regime_alignment_bonus_applied(self, model):
        """Aligned regime should add bonus to win probability."""
        card_aligned = self._make_card("Bull Put Spread", "credit", short_delta=0.25)
        card_misaligned = self._make_card("Bull Put Spread", "credit", short_delta=0.25)

        regime_aligned = self._make_regime("GOLDILOCKS")
        regime_misaligned = self._make_regime("STAGFLATION")

        aligned = model.compute(card_aligned, regime_aligned)
        misaligned = model.compute(card_misaligned, regime_misaligned)

        assert aligned.win_probability > misaligned.win_probability

    def test_high_vol_modifier_reduces_win_prob(self, model):
        """HIGH_VOL modifier should reduce win probability."""
        card_no_mod = self._make_card("Bull Put Spread", "credit", short_delta=0.25)
        card_high_vol = self._make_card("Bull Put Spread", "credit", short_delta=0.25)

        regime_clean = self._make_regime("GOLDILOCKS")
        regime_high_vol = self._make_regime("GOLDILOCKS", modifiers=["HIGH_VOL"])

        clean = model.compute(card_no_mod, regime_clean)
        volatile = model.compute(card_high_vol, regime_high_vol)

        assert clean.win_probability > volatile.win_probability

    def test_win_probability_bounded_zero_to_one(self, model):
        """Win probability must always be in [0, 1]."""
        # Edge case: very high short delta
        card = self._make_card("Bull Put Spread", "credit", short_delta=0.99)
        regime = self._make_regime("STAGFLATION", modifiers=["HIGH_VOL", "TIGHT_LIQ", "VOL_CRUSH_IMMINENT"])
        result = model.compute(card, regime)
        assert 0.0 <= result.win_probability <= 1.0

    def test_win_probability_field_is_set(self, model):
        """After compute(), win_probability should be non-zero."""
        card = self._make_card("Iron Condor", "credit", short_delta=0.20)
        regime = self._make_regime("MIXED")
        result = model.compute(card, regime)
        assert result.win_probability > 0


# ── TradeCard Model Validation ────────────────────────────────────────────────

class TestTradeCardModel:
    """Tests for data/models.TradeCard pydantic model."""

    def _make_valid_card(self, **kwargs) -> TradeCard:
        defaults = dict(
            regime="GOLDILOCKS",
            modifiers=[],
            conviction=65.0,
            transition="stable",
            strategy="Bull Put Spread",
            strategy_type="credit",
            underlying="SPY",
            short_strike=490.0,
            short_delta=0.25,
            long_strike=485.0,
            long_delta=0.12,
            width=5.0,
            expiry="2025-04-18",
            dte=21,
            estimated_credit=1.85,
            max_risk=3.15,
            credit_to_width=0.0,
            breakeven=488.15,
            win_probability=0.68,
            axis_scores={"GROWTH": 35.0, "INFLATION": -10.0, "LIQUIDITY": 15.0, "RISK": 20.0},
            timestamp="2025-03-28T12:00:00",
        )
        defaults.update(kwargs)
        return TradeCard(**defaults)

    def test_credit_to_width_computed(self):
        """credit_to_width should be estimated_credit / width."""
        card = self._make_valid_card(estimated_credit=1.85, width=5.0)
        assert card.credit_to_width == pytest.approx(1.85 / 5.0, rel=0.001)

    def test_all_fields_accessible(self):
        """All required fields should be accessible on the model."""
        card = self._make_valid_card()
        assert card.regime == "GOLDILOCKS"
        assert card.dte == 21
        assert card.underlying == "SPY"
        assert isinstance(card.management, dict)
        assert isinstance(card.notes, list)
