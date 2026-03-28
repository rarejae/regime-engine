"""Tests for engine/regime_classifier.py and engine/axis_scorer.py"""

from __future__ import annotations

import pytest

from data.models import AxisScores, NormalizedIndicator


# ── AxisScorer Tests ──────────────────────────────────────────────────────────

class TestAxisScorer:
    """Tests for engine/axis_scorer.py"""

    @pytest.fixture
    def scorer(self):
        from engine.axis_scorer import AxisScorer
        return AxisScorer()

    def _make_indicator(
        self,
        ind_id: str,
        z_score: float,
        axis: str,
        polarity_sign: int,
        weight: float,
    ) -> NormalizedIndicator:
        return NormalizedIndicator(
            id=ind_id,
            raw_value=0.0,
            z_score=z_score,
            z_score_unclipped=z_score,
            mean=0.0,
            std=1.0,
            polarity_sign=polarity_sign,
            weight=weight,
            axis=axis,
        )

    def test_empty_axis_returns_zero(self, scorer):
        """No indicators on an axis → score=0."""
        scores = scorer.score([])
        assert scores.GROWTH == 0.0
        assert scores.RISK == 0.0

    def test_all_risk_on_gives_positive_score(self, scorer):
        """Indicators all with risk_on polarity and positive z should give positive axis score."""
        indicators = [
            self._make_indicator("A", z_score=2.0, axis="GROWTH", polarity_sign=1, weight=0.5),
            self._make_indicator("B", z_score=1.5, axis="GROWTH", polarity_sign=1, weight=0.5),
        ]
        scores = scorer.score(indicators)
        assert scores.GROWTH > 0

    def test_all_risk_off_negative_z_gives_positive_score(self, scorer):
        """risk_off indicators with negative z (things improving) → positive score."""
        indicators = [
            self._make_indicator("VIX", z_score=-2.0, axis="RISK", polarity_sign=-1, weight=0.30),
        ]
        scores = scorer.score(indicators)
        assert scores.RISK > 0

    def test_ambiguous_indicators_excluded_from_signal(self, scorer):
        """Ambiguous indicators (polarity_sign=0) should not contribute to score."""
        indicators = [
            self._make_indicator("DXY", z_score=3.0, axis="LIQUIDITY", polarity_sign=0, weight=0.20),
            self._make_indicator("NFCI", z_score=-1.5, axis="LIQUIDITY", polarity_sign=-1, weight=0.15),
        ]
        scores = scorer.score(indicators)
        # Only NFCI should contribute; negative z * risk_off(-1) = +1.5 → positive
        assert scores.LIQUIDITY > 0

    def test_output_bounded_to_100(self, scorer):
        """All axis scores should be in [-100, +100]."""
        indicators = [
            self._make_indicator("A", z_score=3.0, axis="INFLATION", polarity_sign=-1, weight=0.5),
            self._make_indicator("B", z_score=3.0, axis="INFLATION", polarity_sign=-1, weight=0.5),
        ]
        scores = scorer.score(indicators)
        assert -100 <= scores.INFLATION <= 100

    def test_coverage_is_1_when_all_present(self, scorer):
        """Coverage should be 1.0 when all axis indicators are available."""
        indicators = [
            self._make_indicator("A", z_score=1.0, axis="RISK", polarity_sign=-1, weight=0.5),
            self._make_indicator("B", z_score=0.5, axis="RISK", polarity_sign=-1, weight=0.5),
        ]
        scores = scorer.score(indicators)
        assert scores.coverage.get("RISK", 0) == pytest.approx(1.0)


# ── RegimeClassifier Tests ────────────────────────────────────────────────────

class TestRegimeClassifier:
    """Tests for engine/regime_classifier.py"""

    @pytest.fixture
    def classifier(self):
        from engine.regime_classifier import RegimeClassifier
        return RegimeClassifier()

    def _make_axis_scores(
        self,
        growth: float = 0.0,
        inflation: float = 0.0,
        liquidity: float = 0.0,
        risk: float = 0.0,
    ) -> AxisScores:
        return AxisScores(
            GROWTH=growth,
            INFLATION=inflation,
            LIQUIDITY=liquidity,
            RISK=risk,
        )

    # Primary regime tests

    def test_goldilocks_regime(self, classifier):
        """Strong growth + low inflation → GOLDILOCKS."""
        scores = self._make_axis_scores(growth=50, inflation=-10)
        result = classifier.classify(scores)
        assert result.primary == "GOLDILOCKS"

    def test_reflation_regime(self, classifier):
        """Strong growth + high inflation → REFLATION."""
        scores = self._make_axis_scores(growth=40, inflation=35)
        result = classifier.classify(scores)
        assert result.primary == "REFLATION"

    def test_stagflation_regime(self, classifier):
        """Weak growth + high inflation → STAGFLATION."""
        scores = self._make_axis_scores(growth=-45, inflation=40)
        result = classifier.classify(scores)
        assert result.primary == "STAGFLATION"

    def test_deflation_regime(self, classifier):
        """Weak growth + falling inflation → DEFLATION."""
        scores = self._make_axis_scores(growth=-50, inflation=-35)
        result = classifier.classify(scores)
        assert result.primary == "DEFLATION"

    def test_mixed_regime_in_transition_zone(self, classifier):
        """Signals near zero → MIXED."""
        scores = self._make_axis_scores(growth=5, inflation=10)
        result = classifier.classify(scores)
        assert result.primary == "MIXED"

    # Modifier tests

    def test_tight_liquidity_modifier(self, classifier):
        """TIGHT_LIQ modifier when liquidity < -30."""
        scores = self._make_axis_scores(growth=30, liquidity=-50)
        result = classifier.classify(scores)
        assert "TIGHT_LIQ" in result.modifiers

    def test_loose_liquidity_modifier(self, classifier):
        """LOOSE_LIQ modifier when liquidity > 30."""
        scores = self._make_axis_scores(growth=30, liquidity=50)
        result = classifier.classify(scores)
        assert "LOOSE_LIQ" in result.modifiers

    def test_high_vol_modifier(self, classifier):
        """HIGH_VOL modifier when risk < -30."""
        scores = self._make_axis_scores(risk=-45)
        result = classifier.classify(scores)
        assert "HIGH_VOL" in result.modifiers

    def test_vol_crush_imminent_modifier(self, classifier):
        """VOL_CRUSH_IMMINENT when vix_term < -3 and vvix > 120."""
        scores = self._make_axis_scores()
        result = classifier.classify(scores, vix_term=-5.0, vvix=130.0)
        assert "VOL_CRUSH_IMMINENT" in result.modifiers

    def test_no_vol_crush_without_threshold(self, classifier):
        """VOL_CRUSH_IMMINENT should NOT trigger when vix_term > -3."""
        scores = self._make_axis_scores()
        result = classifier.classify(scores, vix_term=2.0, vvix=130.0)
        assert "VOL_CRUSH_IMMINENT" not in result.modifiers

    # Conviction tests

    def test_conviction_score_range(self, classifier):
        """Conviction should always be in [0, 100]."""
        for g, i in [(80, 80), (-80, -80), (0, 0), (50, -50)]:
            scores = self._make_axis_scores(growth=g, inflation=i)
            result = classifier.classify(scores)
            assert 0 <= result.conviction <= 100

    def test_high_conviction_for_strong_signals(self, classifier):
        """Strong axis signals should produce high conviction."""
        scores = self._make_axis_scores(growth=80, inflation=70)
        result = classifier.classify(scores)
        assert result.conviction > 50

    def test_low_conviction_for_weak_signals(self, classifier):
        """Weak axis signals should produce low conviction."""
        scores = self._make_axis_scores(growth=5, inflation=-5)
        result = classifier.classify(scores)
        assert result.conviction < 20

    # Transition tests

    def test_first_call_returns_stable(self, classifier):
        """First classification with no history → stable."""
        scores = self._make_axis_scores(growth=40)
        result = classifier.classify(scores)
        assert result.transition == "stable"

    def test_same_regime_increasing_conviction_is_accelerating(self, classifier):
        """Increasing conviction in same regime → accelerating."""
        scores_low = self._make_axis_scores(growth=25, inflation=-5)
        scores_high = self._make_axis_scores(growth=60, inflation=-10)
        classifier.classify(scores_low)
        result = classifier.classify(scores_high)
        assert result.transition == "accelerating"

    def test_regime_change_is_accelerating(self, classifier):
        """Regime change (e.g. GOLDILOCKS → STAGFLATION) → accelerating."""
        classifier.classify(self._make_axis_scores(growth=50, inflation=-10))
        result = classifier.classify(self._make_axis_scores(growth=-40, inflation=35))
        assert result.transition == "accelerating"
        assert result.primary == "STAGFLATION"
