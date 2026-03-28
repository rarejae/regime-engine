"""Strategy scorer — scores all 5 candidate strategies against the full ConditionVector.

Replaces the old 1:1 regime→strategy mapping. Every candidate strategy is scored
against the full ConditionVector. The highest-scoring strategy wins.
No strategy is pre-selected by regime label.
"""

from __future__ import annotations

from loguru import logger

from engine.condition_vector import ConditionVector

STRATEGIES = [
    "bull_put_spread",
    "bull_call_spread",
    "bear_call_spread",
    "bear_put_spread",
    "iron_condor",
]

# Strategy metadata
STRATEGY_TYPE = {
    "bull_put_spread": "credit",
    "bull_call_spread": "debit",
    "bear_call_spread": "credit",
    "bear_put_spread": "debit",
    "iron_condor": "credit",
}

STRATEGY_DISPLAY = {
    "bull_put_spread": "Bull Put Spread",
    "bull_call_spread": "Bull Call Spread",
    "bear_call_spread": "Bear Call Spread",
    "bear_put_spread": "Bear Put Spread",
    "iron_condor": "Iron Condor",
}

OPPOSITE_STRATEGY = {
    "bull_put_spread": "bear_call_spread",
    "bull_call_spread": "bear_put_spread",
    "bear_call_spread": "bull_put_spread",
    "bear_put_spread": "bull_call_spread",
    "iron_condor": "iron_condor",
}


def strategy_type(strategy: str) -> str:
    """Return 'credit' or 'debit' for a strategy."""
    return STRATEGY_TYPE[strategy]


def opposite_strategy(strategy: str) -> str:
    """Return the directional opposite strategy."""
    return OPPOSITE_STRATEGY[strategy]


def score_strategy(strategy: str, cv: ConditionVector) -> float:
    """Score a strategy against the full condition vector.
    Higher = more favorable conditions."""

    if strategy == "bear_call_spread":
        direction_score = (
            -cv.growth * 0.30
            + cv.inflation * 0.20
        )
        vol_edge_score = (
            max(0, -cv.risk) * 0.15
            + max(0, cv.vix - 20) * 0.5
            + max(0, -cv.vix_term) * 2.0
        )
        liquidity_score = (
            -cv.liquidity * 0.10
            + (1 if cv.tight_liq else 0) * 5
        )
        transition_penalty = max(0, cv.growth_delta) * -0.15

        # If vol is already cheap (contango + low VVIX), halve vol edge
        if cv.vix_term > 5 and cv.vvix < 90:
            vol_edge_score *= 0.5

        return direction_score + vol_edge_score + liquidity_score + transition_penalty

    elif strategy == "bull_put_spread":
        direction_score = (
            cv.growth * 0.30
            - cv.inflation * 0.15
        )
        vol_edge_score = (
            max(0, -cv.risk) * 0.10
            + max(0, cv.vix - 15) * 0.3
        )
        # Danger: selling puts into a crash
        if cv.vix > 35:
            vol_edge_score -= (cv.vix - 35) * 0.8

        liquidity_score = (
            cv.liquidity * 0.15
            + (1 if cv.loose_liq else 0) * 5
        )
        transition_bonus = max(0, cv.growth_delta) * 0.10 + max(0, -cv.inflation_delta) * 0.10

        return direction_score + vol_edge_score + liquidity_score + transition_bonus

    elif strategy == "bull_call_spread":
        direction_score = (
            cv.growth * 0.35
            - cv.inflation * 0.10
        )
        # Debit spreads want LOW vol
        vol_cost_score = (
            cv.risk * 0.15
            + max(0, 20 - cv.vix) * 0.5
            + (1 if cv.low_vol else 0) * 8
        )
        if cv.vix > 25:
            vol_cost_score -= (cv.vix - 25) * 0.6

        liquidity_score = cv.liquidity * 0.15
        momentum_bonus = max(0, cv.growth_delta) * 0.20

        return direction_score + vol_cost_score + liquidity_score + momentum_bonus

    elif strategy == "bear_put_spread":
        direction_score = (
            -cv.growth * 0.35
            - cv.inflation * 0.10
        )
        vol_cost_score = (
            max(0, 130 - cv.skew) * 0.3
            + max(0, 20 - cv.vix) * 0.3
        )
        if cv.skew > 140:
            vol_cost_score -= (cv.skew - 140) * 0.5
        if cv.vix > 30:
            vol_cost_score -= (cv.vix - 30) * 0.4

        momentum_bonus = max(0, -cv.growth_delta) * 0.20
        return direction_score + vol_cost_score + momentum_bonus

    elif strategy == "iron_condor":
        direction_penalty = -(abs(cv.growth) + abs(cv.inflation)) * 0.10
        vol_edge_score = (
            max(0, -cv.risk) * 0.20
            + max(0, cv.vix - 18) * 0.4
        )
        stability_bonus = (
            (1.0 - abs(cv.growth_delta) / 50) * 10
            + (1.0 - abs(cv.inflation_delta) / 50) * 10
        )
        if cv.vix_term > 2:
            vol_edge_score += cv.vix_term * 1.5

        return direction_penalty + vol_edge_score + stability_bonus

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def rank_strategies(cv: ConditionVector) -> list[tuple[str, float]]:
    """Score all 5 strategies, return sorted descending. Top 1-3 become trade candidates."""
    scores = [(s, score_strategy(s, cv)) for s in STRATEGIES]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    logger.info("Strategy rankings:")
    for i, (s, sc) in enumerate(ranked):
        marker = " <-- TOP" if i == 0 else ""
        logger.info(f"  #{i+1} {STRATEGY_DISPLAY[s]:20} score={sc:+.2f}{marker}")

    return ranked


def select_short_delta(cv: ConditionVector, strategy: str) -> float:
    """Delta selection uses the RELEVANT axis magnitudes per strategy direction."""
    if strategy in ("bear_call_spread", "bear_put_spread"):
        primary_signal = abs(cv.growth)
        secondary_signal = cv.inflation
    elif strategy in ("bull_put_spread", "bull_call_spread"):
        primary_signal = abs(cv.growth)
        secondary_signal = -cv.inflation
    else:
        primary_signal = abs(cv.risk)
        secondary_signal = 0

    effective_conviction = primary_signal * 0.7 + max(0, secondary_signal) * 0.3

    if effective_conviction > 60:
        base_delta = 0.30 + (effective_conviction - 60) / 200
    elif effective_conviction > 30:
        base_delta = 0.20 + (effective_conviction - 30) / 150
    else:
        base_delta = 0.12 + effective_conviction / 250

    # SKEW adjustment for put strategies
    if strategy in ("bull_put_spread", "bear_put_spread"):
        if cv.skew > 140:
            base_delta -= 0.03
        elif cv.skew < 120:
            base_delta += 0.02

    # Transition momentum adjustment
    if strategy in ("bear_call_spread", "bear_put_spread"):
        if cv.growth_delta < -5:
            base_delta += 0.02
    elif strategy in ("bull_put_spread", "bull_call_spread"):
        if cv.growth_delta > 5:
            base_delta += 0.02

    return round(min(0.40, max(0.08, base_delta)), 2)


def select_dte(cv: ConditionVector, strat_type: str) -> int:
    """DTE driven by liquidity, vol term structure, and transition rate."""
    base_dte = 35  # default

    # Liquidity: tight = shorter exposure window
    if cv.tight_liq:
        base_dte -= 10
    elif cv.loose_liq:
        base_dte += 10

    # Vol term structure: backwardation = sell near-dated, contango = go further out
    if cv.vix_term < -2:
        base_dte -= 7
    elif cv.vix_term > 4:
        base_dte += 7

    # Credit vs debit: credit spreads harvest theta faster at shorter DTE
    if strat_type == "credit":
        base_dte -= 5
    else:
        base_dte += 10

    # Transition speed: fast-moving regimes = shorter DTE
    transition_speed = abs(cv.growth_delta) + abs(cv.inflation_delta)
    if transition_speed > 30:
        base_dte -= 7

    return max(14, min(75, base_dte))
