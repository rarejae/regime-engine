"""ConditionVector — the central data structure driving all trade decisions.

The ConditionVector captures the full macro state: continuous axis scores,
transition directions, raw vol surface data, credit conditions, and modifier flags.

The regime label is for DISPLAY ONLY. All strategy scoring, strike selection,
DTE, and width decisions flow from the ConditionVector fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from data.models import AxisScores, IndicatorSnapshot


@dataclass
class ConditionVector:
    """The full macro state that drives every trade decision.
    Nothing is collapsed into a label before reaching the strategy engine."""

    # Axis scores (continuous, -100 to +100)
    growth: float
    inflation: float
    liquidity: float
    risk: float

    # Transition directions (rate of change of each axis over trailing 30 days)
    growth_delta: float        # positive = improving, negative = deteriorating
    inflation_delta: float     # positive = heating up, negative = cooling
    liquidity_delta: float     # positive = loosening, negative = tightening
    risk_delta: float          # positive = calming, negative = stress increasing

    # Raw vol surface (not summarized — these drive spread structure directly)
    vix: float
    vix_term: float            # VIX3M - VIX
    vvix: float
    skew: float
    move_index: float

    # Credit conditions (raw, not just the axis score)
    hy_oas: float
    ig_oas: float
    hy_ig_diff: float

    # Modifier flags (derived but kept as booleans for rule gating)
    tight_liq: bool
    loose_liq: bool
    high_vol: bool
    low_vol: bool
    vol_crush_imminent: bool

    # Regime label (for DISPLAY ONLY — NOT used in scoring)
    regime_label: str          # "STAGFLATION", "GOLDILOCKS", etc.
    conviction: float          # 0-100


def build_condition_vector(
    axis_scores: AxisScores,
    snapshot: IndicatorSnapshot,
    prev_axis_scores: Optional[AxisScores] = None,
    regime_label: str = "MIXED",
    conviction: float = 50.0,
) -> ConditionVector:
    """Build a ConditionVector from axis scores and indicator snapshot.

    Args:
        axis_scores: Current axis scores from AxisScorer.
        snapshot: Current indicator snapshot with raw values.
        prev_axis_scores: Axis scores from ~30 days ago (for delta computation).
            If None, deltas default to 0.
        regime_label: Display-only regime label from RegimeClassifier.
        conviction: Conviction score from RegimeClassifier.

    Returns:
        Fully populated ConditionVector.
    """

    def _get(indicator_id: str, default: float = 0.0) -> float:
        iv = snapshot.get(indicator_id)
        return iv.value if iv is not None else default

    # Transition directions
    if prev_axis_scores is not None:
        growth_delta = axis_scores.GROWTH - prev_axis_scores.GROWTH
        inflation_delta = axis_scores.INFLATION - prev_axis_scores.INFLATION
        liquidity_delta = axis_scores.LIQUIDITY - prev_axis_scores.LIQUIDITY
        risk_delta = axis_scores.RISK - prev_axis_scores.RISK
    else:
        growth_delta = 0.0
        inflation_delta = 0.0
        liquidity_delta = 0.0
        risk_delta = 0.0

    # Raw vol surface
    vix = _get("VIX", 20.0)
    vix_term = _get("VIX_TERM", 0.0)
    vvix = _get("VVIX", 100.0)
    skew = _get("SKEW", 130.0)
    move_index = _get("MOVE_INDEX", 100.0)

    # Credit conditions
    hy_oas = _get("HY_OAS", 400.0)
    ig_oas = _get("IG_OAS", 100.0)
    hy_ig_diff = _get("HY_IG_DIFF", 300.0)

    # Modifier flags
    liq = axis_scores.LIQUIDITY
    rsk = axis_scores.RISK
    tight_liq = liq < -30
    loose_liq = liq > 30
    high_vol = rsk < -30
    low_vol = rsk > 30
    vol_crush_imminent = (vix_term < -3.0 and vvix > 120)

    cv = ConditionVector(
        growth=axis_scores.GROWTH,
        inflation=axis_scores.INFLATION,
        liquidity=axis_scores.LIQUIDITY,
        risk=axis_scores.RISK,
        growth_delta=growth_delta,
        inflation_delta=inflation_delta,
        liquidity_delta=liquidity_delta,
        risk_delta=risk_delta,
        vix=vix,
        vix_term=vix_term,
        vvix=vvix,
        skew=skew,
        move_index=move_index,
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        hy_ig_diff=hy_ig_diff,
        tight_liq=tight_liq,
        loose_liq=loose_liq,
        high_vol=high_vol,
        low_vol=low_vol,
        vol_crush_imminent=vol_crush_imminent,
        regime_label=regime_label,
        conviction=conviction,
    )

    logger.info(
        f"ConditionVector built | G:{cv.growth:+.0f} I:{cv.inflation:+.0f} "
        f"L:{cv.liquidity:+.0f} R:{cv.risk:+.0f} | "
        f"VIX:{cv.vix:.1f} term:{cv.vix_term:.1f} VVIX:{cv.vvix:.0f} | "
        f"label:{cv.regime_label} (display only)"
    )
    return cv
