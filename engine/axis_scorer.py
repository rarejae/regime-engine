"""Axis composite scorer for the Macro Regime Engine.

Takes the list of NormalizedIndicator objects and computes a single composite
score for each of the four axes: GROWTH, INFLATION, LIQUIDITY, RISK.

Scoring formula:
    raw = sum(z_score * weight * polarity_sign)
    axis_score = tanh(raw) * 100  → bounded in [-100, +100]

Indicators with polarity="ambiguous" (sign=0) contribute only through their
weight normalization, not their directional signal.
"""

from __future__ import annotations

import os

import numpy as np
import yaml
from loguru import logger

from data.models import AxisScores, NormalizedIndicator

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

_TANH_SCALE = _SETTINGS["axis_scoring"]["tanh_scale"]

AXES = ("GROWTH", "INFLATION", "LIQUIDITY", "RISK")


class AxisScorer:
    """Computes composite axis scores from normalized indicator z-scores."""

    def score(self, normalized: list[NormalizedIndicator]) -> AxisScores:
        """
        Compute all four axis scores from the list of normalized indicators.

        Args:
            normalized: List of NormalizedIndicator objects from the Normalizer.

        Returns:
            AxisScores with GROWTH, INFLATION, LIQUIDITY, RISK in [-100, +100].
        """
        scores: dict[str, float] = {}
        coverage: dict[str, float] = {}

        for axis in AXES:
            axis_indicators = [n for n in normalized if n.axis == axis]
            score, cov = self._compute_axis(axis, axis_indicators)
            scores[axis] = score
            coverage[axis] = cov

        result = AxisScores(
            GROWTH=scores["GROWTH"],
            INFLATION=scores["INFLATION"],
            LIQUIDITY=scores["LIQUIDITY"],
            RISK=scores["RISK"],
            coverage=coverage,
        )

        logger.info(
            f"Axis scores — "
            f"GROWTH:{result.GROWTH:+.1f}  "
            f"INFLATION:{result.INFLATION:+.1f}  "
            f"LIQUIDITY:{result.LIQUIDITY:+.1f}  "
            f"RISK:{result.RISK:+.1f}"
        )
        return result

    @staticmethod
    def _compute_axis(axis: str, indicators: list[NormalizedIndicator]) -> tuple[float, float]:
        """
        Compute the tanh-bounded composite score for one axis.

        Args:
            axis: Axis name (for logging).
            indicators: All NormalizedIndicator objects belonging to this axis.

        Returns:
            Tuple of (axis_score: float in [-100, +100], coverage: float in [0, 1]).
        """
        if not indicators:
            logger.warning(f"No indicators available for axis {axis}; score=0.0")
            return 0.0, 0.0

        # Only include indicators with a directional signal (polarity_sign != 0)
        directional = [n for n in indicators if n.polarity_sign != 0]

        if not directional:
            logger.warning(f"All {axis} indicators are ambiguous; score=0.0")
            return 0.0, 0.0

        total_weight = sum(n.weight for n in directional)
        if total_weight == 0:
            return 0.0, 0.0

        # Proportionally renormalize weights to handle missing indicators
        raw = 0.0
        for n in directional:
            normalized_weight = n.weight / total_weight
            contribution = n.z_score * normalized_weight * n.polarity_sign
            raw += contribution
            logger.debug(
                f"  {axis} | {n.id:20} z={n.z_score:>6.3f} "
                f"w={normalized_weight:.3f} sign={n.polarity_sign:+d} → {contribution:+.4f}"
            )

        # Max possible weight (all configured indicators present)
        configured_total = sum(n.weight for n in indicators)  # includes ambiguous
        coverage = total_weight / configured_total if configured_total > 0 else 0.0

        # Tanh bounding to [-100, +100]
        axis_score = float(np.tanh(raw * _TANH_SCALE) * 100)

        logger.debug(f"  {axis} raw={raw:.4f} → score={axis_score:.1f} (coverage={coverage:.0%})")
        return axis_score, coverage
