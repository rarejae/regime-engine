"""Strategy selector for the Macro Regime Engine.

Maps the classified regime to the appropriate vertical spread strategy type
and parameters, drawing from regimes.yaml strategy_map.
"""

from __future__ import annotations

import os
from typing import Any

import yaml
from loguru import logger

from data.models import RegimeClassification

_REGIMES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "regimes.yaml")
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")

with open(_REGIMES_PATH) as _f:
    _REGIMES_CFG = yaml.safe_load(_f)

with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

_STRATEGY_MAP: dict[str, dict] = _REGIMES_CFG["strategy_map"]
_MANAGEMENT: dict[str, dict] = _REGIMES_CFG["management"]
_CONVICTION_CFG = _SETTINGS["regime"]
_STRIKE_CFG = _SETTINGS["strike_selection"]
_DTE_CFG = _SETTINGS["dte_selection"]
_WIDTH_CFG = _SETTINGS["spread_width"]


class StrategySelector:
    """
    Selects the optimal vertical spread strategy from a regime classification.
    """

    def select(self, regime: RegimeClassification, vix: float) -> dict[str, Any]:
        """
        Map a regime classification to strategy parameters.

        Args:
            regime: RegimeClassification from RegimeClassifier.
            vix: Current VIX level (used for spread width determination).

        Returns:
            Dict with keys: strategy, strategy_type, direction, delta_targets,
            dte, width_range, management, rationale.
        """
        primary = regime.primary
        strategy_cfg = _STRATEGY_MAP.get(primary, _STRATEGY_MAP["MIXED"])

        conviction = regime.conviction
        strategy_type = strategy_cfg["type"]

        delta_targets = self._delta_targets(conviction)
        dte = self._dte(conviction, strategy_type)
        width_range = self._width_range(vix)
        management = _MANAGEMENT.get(
            f"{strategy_type}_spreads" if strategy_type != "credit" or "condor" not in strategy_cfg["strategy"].lower()
            else "iron_condor",
            _MANAGEMENT.get("credit_spreads", {}),
        )

        # Iron Condor gets its own management block
        if "Iron Condor" in strategy_cfg["strategy"]:
            management = _MANAGEMENT.get("iron_condor", management)

        result = {
            "strategy": strategy_cfg["strategy"],
            "strategy_type": strategy_type,
            "direction": strategy_cfg["direction"],
            "delta_targets": delta_targets,
            "dte": dte,
            "width_range": width_range,
            "management": management,
            "rationale": strategy_cfg.get("rationale", ""),
            "modifiers": regime.modifiers,
        }

        logger.info(
            f"Strategy selected: {result['strategy']} | DTE: {dte} | "
            f"Short Δ: {delta_targets['short_min']:.2f}–{delta_targets['short_max']:.2f} | "
            f"Width: ${width_range['min']}–${width_range['max']}"
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _delta_targets(self, conviction: float) -> dict[str, float]:
        """
        Return short and long delta target ranges based on conviction.

        Args:
            conviction: Conviction score 0–100.

        Returns:
            Dict with short_min, short_max, long_min, long_max.
        """
        high = _CONVICTION_CFG["conviction_high"]
        medium = _CONVICTION_CFG["conviction_medium"]

        if conviction > high:
            cfg = _STRIKE_CFG["high_conviction"]
        elif conviction > medium:
            cfg = _STRIKE_CFG["medium_conviction"]
        else:
            cfg = _STRIKE_CFG["low_conviction"]

        return {
            "short_min": cfg["short_delta_min"],
            "short_max": cfg["short_delta_max"],
            "long_min": cfg["long_delta_min"],
            "long_max": cfg["long_delta_max"],
        }

    @staticmethod
    def _dte(conviction: float, strategy_type: str) -> int:
        """
        Select target DTE based on conviction level and strategy type.

        Args:
            conviction: Conviction score 0–100.
            strategy_type: "credit" or "debit".

        Returns:
            Target DTE as int.
        """
        type_key = "credit" if strategy_type == "credit" else "debit"
        dte_cfg = _DTE_CFG[type_key]

        high = _CONVICTION_CFG["conviction_high"]
        medium = _CONVICTION_CFG["conviction_medium"]

        if conviction > high:
            return dte_cfg["high_conviction"]
        elif conviction > medium:
            return dte_cfg["medium_conviction"]
        else:
            return dte_cfg["low_conviction"]

    @staticmethod
    def _width_range(vix: float) -> dict[str, int]:
        """
        Select spread width range (in SPY points) based on VIX regime.

        Args:
            vix: Current VIX level.

        Returns:
            Dict with 'min' and 'max' spread width values.
        """
        if vix < 15:
            return _WIDTH_CFG["vix_lt_15"]
        elif vix < 25:
            return _WIDTH_CFG["vix_15_25"]
        elif vix < 40:
            return _WIDTH_CFG["vix_25_40"]
        else:
            return _WIDTH_CFG["vix_gt_40"]
