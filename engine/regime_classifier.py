"""Regime classifier for the Macro Regime Engine.

Classifies the current macroeconomic environment into a primary regime
(GOLDILOCKS, REFLATION, STAGFLATION, DEFLATION, MIXED) plus modifier tags,
and computes a conviction score.

NOTE: The regime label is for DISPLAY ONLY. All trade decisions flow from
the ConditionVector (engine/condition_vector.py), not the regime label.

Also tracks regime transition direction by comparing to the previous classification.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml
from loguru import logger

from data.models import AxisScores, RegimeClassification

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
_REGIMES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "regimes.yaml")

with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

with open(_REGIMES_PATH) as _f:
    _REGIMES_CFG = yaml.safe_load(_f)

_REGIME_CFG = _SETTINGS["regime"]
_PRIMARY_CFG = _REGIMES_CFG["primary"]
_MODIFIER_CFG = _REGIMES_CFG["modifiers"]


class RegimeClassifier:
    """
    Classifies macroeconomic regime from axis scores.

    Maintains a short history of recent classifications to detect transition
    direction (accelerating / decelerating / stable).
    """

    def __init__(self) -> None:
        self._history: list[RegimeClassification] = []

    def classify(self, axis_scores: AxisScores, vix_term: Optional[float] = None, vvix: Optional[float] = None) -> RegimeClassification:
        """
        Classify the current regime from axis scores.

        Args:
            axis_scores: Computed axis scores from AxisScorer.
            vix_term: VIX term structure spread (VIX3M - VIX). Optional.
            vvix: Current VVIX level. Optional.

        Returns:
            RegimeClassification with primary regime, modifiers, and conviction.
        """
        threshold = float(_PRIMARY_CFG["growth_threshold"])
        infl_threshold = float(_PRIMARY_CFG["inflation_threshold"])

        primary = self._classify_primary(
            axis_scores.GROWTH,
            axis_scores.INFLATION,
            threshold,
            infl_threshold,
        )

        modifiers = self._compute_modifiers(
            axis_scores.LIQUIDITY,
            axis_scores.RISK,
            vix_term,
            vvix,
        )

        conviction = self._compute_conviction(axis_scores.GROWTH, axis_scores.INFLATION)
        transition = self._compute_transition(primary, conviction)

        classification = RegimeClassification(
            primary=primary,
            modifiers=modifiers,
            conviction=round(conviction, 1),
            transition=transition,
            axis_scores=axis_scores,
        )

        self._history.append(classification)
        if len(self._history) > 10:
            self._history.pop(0)

        logger.info(
            f"Regime: {primary} | Conviction: {conviction:.0f} | "
            f"Modifiers: {modifiers or 'none'} | Transition: {transition}"
        )
        return classification

    # ── Private methods ────────────────────────────────────────────────────────

    @staticmethod
    def _classify_primary(
        growth: float,
        inflation: float,
        growth_threshold: float,
        infl_threshold: float,
    ) -> str:
        """
        Classify the primary regime from growth and inflation axis scores.

        Args:
            growth: Growth axis score in [-100, +100].
            inflation: Inflation axis score in [-100, +100].
            growth_threshold: Threshold for "expansion" vs "contraction" signal.
            infl_threshold: Threshold for "elevated" vs "subdued" inflation signal.

        Returns:
            Primary regime string.
        """
        if growth > growth_threshold and inflation < infl_threshold:
            return "GOLDILOCKS"
        elif growth > growth_threshold and inflation >= infl_threshold:
            return "REFLATION"
        elif growth <= -growth_threshold and inflation >= infl_threshold:
            return "STAGFLATION"
        elif growth <= -growth_threshold and inflation < -infl_threshold:
            return "DEFLATION"
        else:
            return "MIXED"

    @staticmethod
    def _compute_modifiers(
        liquidity: float,
        risk: float,
        vix_term: Optional[float],
        vvix: Optional[float],
    ) -> list[str]:
        """
        Compute modifier tags from liquidity, risk axis scores and vol signals.

        Args:
            liquidity: Liquidity axis score.
            risk: Risk axis score.
            vix_term: VIX3M - VIX spread (positive = contango = calm).
            vvix: VVIX level.

        Returns:
            List of modifier strings.
        """
        mods: list[str] = []

        if liquidity < -30:
            mods.append("TIGHT_LIQ")
        elif liquidity > 30:
            mods.append("LOOSE_LIQ")

        if risk < -30:
            mods.append("HIGH_VOL")
        elif risk > 30:
            mods.append("LOW_VOL")

        # Vol crush imminent: inverted term structure + very high vol-of-vol
        if vix_term is not None and vvix is not None:
            if vix_term < -3.0 and vvix > 120:
                mods.append("VOL_CRUSH_IMMINENT")

        return mods

    @staticmethod
    def _compute_conviction(growth: float, inflation: float) -> float:
        """
        Compute conviction score as the average absolute axis signal strength.

        Args:
            growth: Growth axis score.
            inflation: Inflation axis score.

        Returns:
            Conviction score in [0, 100].
        """
        return (abs(growth) + abs(inflation)) / 2

    def _compute_transition(self, current_regime: str, current_conviction: float) -> str:
        """
        Determine whether the regime signal is accelerating, decelerating, or stable.

        Compares current conviction to the most recent prior classification.

        Args:
            current_regime: The just-classified primary regime.
            current_conviction: Current conviction score.

        Returns:
            "accelerating" | "decelerating" | "stable"
        """
        if not self._history:
            return "stable"

        prev = self._history[-1]

        if prev.primary != current_regime:
            return "accelerating"  # regime just changed = new trend forming

        delta = current_conviction - prev.conviction
        if delta > 5:
            return "accelerating"
        elif delta < -5:
            return "decelerating"
        else:
            return "stable"
