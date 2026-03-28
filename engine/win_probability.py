"""Win probability model for the Macro Regime Engine.

Combines delta-implied win probability with regime alignment adjustments
to produce a regime-aware estimate of trade success probability.

For credit spreads: base = 1 - abs(short_delta)
For debit spreads:  base = abs(long_delta)  (need to reach long strike)
"""

from __future__ import annotations

import os
from typing import Any

import yaml
from loguru import logger

from data.models import RegimeClassification, TradeCard

_REGIMES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "regimes.yaml")
with open(_REGIMES_PATH) as _f:
    _REGIMES_CFG = yaml.safe_load(_f)

_WIN_CFG = _REGIMES_CFG["win_probability"]

# Which regimes are "aligned" with which trade directions
_BULLISH_REGIMES = {"GOLDILOCKS", "REFLATION"}
_BEARISH_REGIMES = {"STAGFLATION", "DEFLATION"}
_NEUTRAL_REGIMES = {"MIXED"}

_BULLISH_STRATEGIES = {"Bull Put Spread", "Bull Call Spread"}
_BEARISH_STRATEGIES = {"Bear Call Spread", "Bear Put Spread"}
_NEUTRAL_STRATEGIES = {"Iron Condor"}


class WinProbabilityModel:
    """Computes regime-adjusted win probability for a TradeCard."""

    def compute(
        self,
        card: TradeCard,
        regime: RegimeClassification,
    ) -> TradeCard:
        """
        Compute and attach win_probability to a TradeCard.

        Args:
            card: TradeCard with strikes and delta already filled in.
            regime: RegimeClassification used to apply alignment bonus.

        Returns:
            Updated TradeCard with win_probability set.
        """
        base = self._delta_implied_base(card)
        adjustment = self._regime_adjustment(card, regime)

        win_prob = max(0.0, min(1.0, base + adjustment))

        logger.info(
            f"Win probability: base={base:.3f} "
            f"adj={adjustment:+.3f} → final={win_prob:.3f} "
            f"({win_prob*100:.1f}%)"
        )

        return card.model_copy(update={"win_probability": round(win_prob, 4)})

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _delta_implied_base(card: TradeCard) -> float:
        """
        Compute the delta-implied base win probability.

        Credit spreads: probability of expiring OTM ≈ 1 - |short_delta|
        Debit spreads: probability of reaching max value ≈ |long_delta|

        Args:
            card: TradeCard with short_delta, long_delta, strategy_type set.

        Returns:
            Base win probability in [0, 1].
        """
        if card.strategy_type == "credit":
            # For credit: win if short strike expires OTM
            # Calls: P(SPY < short_strike) ≈ 1 - short_delta
            # Puts:  P(SPY > short_strike) ≈ 1 - |short_delta|
            return 1.0 - abs(card.short_delta)
        else:
            # For debit: win if spread fully in-the-money at expiry
            # Calls: P(SPY > long_strike) ≈ long_delta
            # Puts:  P(SPY < long_strike) ≈ |long_delta|
            return abs(card.long_delta)

    @staticmethod
    def _regime_adjustment(card: TradeCard, regime: RegimeClassification) -> float:
        """
        Compute the regime alignment adjustment to win probability.

        Applies:
        - Bonus when regime strongly aligns with trade direction
        - Penalties for HIGH_VOL, TIGHT_LIQ, VOL_CRUSH_IMMINENT modifiers
        - Small bonus for LOW_VOL, LOOSE_LIQ

        Args:
            card: TradeCard with strategy name and modifiers.
            regime: RegimeClassification.

        Returns:
            Adjustment as float (can be positive or negative).
        """
        adjustment = 0.0
        strategy = card.strategy
        primary = regime.primary

        # Alignment bonus
        regime_bonus = float(_WIN_CFG.get("regime_alignment_bonus", 0.05))
        if strategy in _BULLISH_STRATEGIES and primary in _BULLISH_REGIMES:
            adjustment += regime_bonus
        elif strategy in _BEARISH_STRATEGIES and primary in _BEARISH_REGIMES:
            adjustment += regime_bonus
        elif strategy in _NEUTRAL_STRATEGIES and primary in _NEUTRAL_REGIMES:
            adjustment += regime_bonus

        # Modifier adjustments
        modifier_penalties: dict[str, float] = _WIN_CFG.get("modifier_penalties", {})
        modifier_bonuses: dict[str, float] = _WIN_CFG.get("modifier_bonuses", {})

        for mod in regime.modifiers:
            if mod in modifier_penalties:
                pen = float(modifier_penalties[mod])
                adjustment -= pen
                logger.debug(f"Win prob penalty: {mod} → -{pen:.3f}")
            if mod in modifier_bonuses:
                bon = float(modifier_bonuses[mod])
                adjustment += bon
                logger.debug(f"Win prob bonus:   {mod} → +{bon:.3f}")

        return adjustment
