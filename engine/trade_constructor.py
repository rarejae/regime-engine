"""Trade constructor for the Macro Regime Engine.

Takes a strategy selection dict and builds a concrete TradeCard with specific
SPY strikes, expiry, width, estimated credit/debit, and breakeven.

Uses the live SPY options chain from yfinance + Black-Scholes delta computation.
Handles real-world messiness: filters illiquid strikes, wide bid-ask, zero OI.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from data.models import RegimeClassification, TradeCard
from utils.helpers import (
    filter_chain,
    find_strike_by_delta,
    mid_price,
    target_expiry_date,
    years_to_expiry,
)

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
_REGIMES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "regimes.yaml")

with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

with open(_REGIMES_PATH) as _f:
    _REGIMES_CFG = yaml.safe_load(_f)

_OPT_FILTERS = _SETTINGS["options_filters"]


class TradeConstructor:
    """Constructs a concrete SPY vertical spread trade recommendation."""

    def __init__(self, fetcher: Any) -> None:
        """
        Args:
            fetcher: data.fetcher.Fetcher instance (for spot price, RF rate, options).
        """
        self.fetcher = fetcher

    def construct(
        self,
        regime: RegimeClassification,
        strategy_params: dict[str, Any],
    ) -> TradeCard:
        """
        Build a complete TradeCard for the current regime and strategy.

        Args:
            regime: RegimeClassification from RegimeClassifier.
            strategy_params: Output dict from StrategySelector.select().

        Returns:
            TradeCard with all trade details filled in.
        """
        from data.sources.yahoo import YahooSource

        yahoo = YahooSource()
        S = self.fetcher.get_spy_spot()
        r = self.fetcher.get_risk_free_rate()
        strategy = strategy_params["strategy"]
        strategy_type = strategy_params["strategy_type"]
        delta_targets = strategy_params["delta_targets"]
        target_dte = strategy_params["dte"]
        width_range = strategy_params["width_range"]

        # ── Get options chain ──────────────────────────────────────────────────
        expirations, spy_ticker = yahoo.get_options_chain("SPY")
        expiry = target_expiry_date(target_dte, expirations)
        T = years_to_expiry(expiry)
        actual_dte = int(T * 365)

        logger.info(f"Constructing {strategy} | SPY spot={S:.2f} | expiry={expiry} ({actual_dte}d) | T={T:.4f}yr")

        chain_calls, chain_puts = yahoo.get_chain_for_expiry(expiry, "SPY")

        # ── Route to the right constructor ────────────────────────────────────
        if strategy == "Bull Put Spread":
            trade = self._build_bull_put_spread(
                S, T, r, chain_puts, delta_targets, width_range
            )
        elif strategy == "Bear Call Spread":
            trade = self._build_bear_call_spread(
                S, T, r, chain_calls, delta_targets, width_range
            )
        elif strategy == "Bull Call Spread":
            trade = self._build_bull_call_spread(
                S, T, r, chain_calls, delta_targets, width_range
            )
        elif strategy == "Bear Put Spread":
            trade = self._build_bear_put_spread(
                S, T, r, chain_puts, delta_targets, width_range
            )
        elif strategy == "Iron Condor":
            trade = self._build_iron_condor(
                S, T, r, chain_calls, chain_puts, delta_targets, width_range
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # ── Collect notes ──────────────────────────────────────────────────────
        notes: list[str] = []
        if regime.modifiers:
            notes.append(f"Active modifiers: {', '.join(regime.modifiers)}")
        for ind_id, iv in (self.fetcher.config if hasattr(self.fetcher, "config") else {}).items():
            pass  # proxy notes collected from snapshot warnings at caller level

        # ── Assemble TradeCard ─────────────────────────────────────────────────
        card = TradeCard(
            regime=regime.primary,
            modifiers=regime.modifiers,
            conviction=regime.conviction,
            transition=regime.transition,
            strategy=strategy,
            strategy_type=strategy_type,
            underlying="SPY",
            short_strike=trade["short_strike"],
            short_delta=trade["short_delta"],
            long_strike=trade["long_strike"],
            long_delta=trade["long_delta"],
            width=trade["width"],
            expiry=expiry,
            dte=actual_dte,
            estimated_credit=trade["estimated_credit"],
            max_risk=trade["max_risk"],
            credit_to_width=0.0,  # computed by model_validator
            breakeven=trade["breakeven"],
            win_probability=0.0,  # filled by WinProbabilityModel
            axis_scores={
                "GROWTH": regime.axis_scores.GROWTH,
                "INFLATION": regime.axis_scores.INFLATION,
                "LIQUIDITY": regime.axis_scores.LIQUIDITY,
                "RISK": regime.axis_scores.RISK,
            },
            management=strategy_params.get("management", {}),
            notes=notes,
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"TradeCard: {strategy} SPY {card.short_strike}/{card.long_strike} "
            f"exp={expiry} credit=${card.estimated_credit:.2f} "
            f"max_risk=${card.max_risk:.2f}"
        )
        return card

    # ── Strategy builders ──────────────────────────────────────────────────────

    def _build_bull_put_spread(
        self,
        S: float,
        T: float,
        r: float,
        puts: pd.DataFrame,
        delta_targets: dict,
        width_range: dict,
    ) -> dict:
        """
        Bull Put Spread: sell OTM put, buy further OTM put.
        Direction: bullish (profit if SPY stays above short strike).
        """
        puts_filtered = filter_chain(
            puts,
            min_open_interest=_OPT_FILTERS["min_open_interest"],
            max_bid_ask_spread_pct=_OPT_FILTERS["max_bid_ask_spread_pct"],
            min_volume=_OPT_FILTERS["min_volume"],
        )

        # Short put: target_delta (negative for put; pass magnitude)
        short_delta_target = (delta_targets["short_min"] + delta_targets["short_max"]) / 2
        short_row = find_strike_by_delta(puts_filtered, short_delta_target, S, T, r, "put")
        if short_row is None:
            short_row = self._nearest_otm_put(puts_filtered, S, 0.95)

        short_strike = float(short_row["strike"])
        short_iv = float(short_row.get("impliedVolatility", 0.20) or 0.20)
        from utils.helpers import bs_delta as _bs_delta
        short_delta_actual = _bs_delta(S, short_strike, T, r, short_iv, "put")

        # Long put: further OTM by spread width
        target_width = (width_range["min"] + width_range["max"]) / 2
        long_strike = round(short_strike - target_width, 0)
        long_delta_target = (delta_targets["long_min"] + delta_targets["long_max"]) / 2
        long_row = find_strike_by_delta(puts_filtered, long_delta_target, S, T, r, "put")
        if long_row is not None and abs(float(long_row["strike"]) - short_strike) >= 1:
            long_strike = float(long_row["strike"])
        long_iv = float(long_row.get("impliedVolatility", 0.20) or 0.20) if long_row is not None else short_iv
        long_delta_actual = _bs_delta(S, long_strike, T, r, long_iv, "put")

        short_credit = mid_price(short_row)
        long_cost = self._lookup_strike_mid(puts_filtered, long_strike) or mid_price(long_row) if long_row is not None else 0.0
        net_credit = max(short_credit - long_cost, 0.01)
        width = abs(short_strike - long_strike)
        max_risk = width - net_credit
        breakeven = short_strike - net_credit

        return {
            "short_strike": short_strike,
            "short_delta": abs(short_delta_actual),
            "long_strike": long_strike,
            "long_delta": abs(long_delta_actual),
            "width": width,
            "estimated_credit": round(net_credit, 2),
            "max_risk": round(max_risk, 2),
            "breakeven": round(breakeven, 2),
        }

    def _build_bear_call_spread(
        self,
        S: float,
        T: float,
        r: float,
        calls: pd.DataFrame,
        delta_targets: dict,
        width_range: dict,
    ) -> dict:
        """
        Bear Call Spread: sell OTM call, buy further OTM call.
        Direction: bearish (profit if SPY stays below short strike).
        """
        calls_filtered = filter_chain(
            calls,
            min_open_interest=_OPT_FILTERS["min_open_interest"],
            max_bid_ask_spread_pct=_OPT_FILTERS["max_bid_ask_spread_pct"],
            min_volume=_OPT_FILTERS["min_volume"],
        )

        short_delta_target = (delta_targets["short_min"] + delta_targets["short_max"]) / 2
        short_row = find_strike_by_delta(calls_filtered, short_delta_target, S, T, r, "call")
        if short_row is None:
            short_row = self._nearest_otm_call(calls_filtered, S, 1.05)

        short_strike = float(short_row["strike"])
        short_iv = float(short_row.get("impliedVolatility", 0.20) or 0.20)
        from utils.helpers import bs_delta as _bs_delta
        short_delta_actual = _bs_delta(S, short_strike, T, r, short_iv, "call")

        target_width = (width_range["min"] + width_range["max"]) / 2
        long_delta_target = (delta_targets["long_min"] + delta_targets["long_max"]) / 2
        long_row = find_strike_by_delta(calls_filtered, long_delta_target, S, T, r, "call")
        long_strike = float(long_row["strike"]) if long_row is not None else short_strike + target_width
        long_iv = float(long_row.get("impliedVolatility", 0.20) or 0.20) if long_row is not None else short_iv
        long_delta_actual = _bs_delta(S, long_strike, T, r, long_iv, "call")

        short_credit = mid_price(short_row)
        long_cost = self._lookup_strike_mid(calls_filtered, long_strike) or (mid_price(long_row) if long_row is not None else 0.0)
        net_credit = max(short_credit - long_cost, 0.01)
        width = abs(long_strike - short_strike)
        max_risk = width - net_credit
        breakeven = short_strike + net_credit

        return {
            "short_strike": short_strike,
            "short_delta": short_delta_actual,
            "long_strike": long_strike,
            "long_delta": long_delta_actual,
            "width": width,
            "estimated_credit": round(net_credit, 2),
            "max_risk": round(max_risk, 2),
            "breakeven": round(breakeven, 2),
        }

    def _build_bull_call_spread(
        self,
        S: float,
        T: float,
        r: float,
        calls: pd.DataFrame,
        delta_targets: dict,
        width_range: dict,
    ) -> dict:
        """
        Bull Call Spread: buy ATM/slightly-OTM call, sell further OTM call.
        Direction: bullish, debit strategy.
        """
        calls_filtered = filter_chain(
            calls,
            min_open_interest=_OPT_FILTERS["min_open_interest"],
            max_bid_ask_spread_pct=_OPT_FILTERS["max_bid_ask_spread_pct"],
            min_volume=_OPT_FILTERS["min_volume"],
        )
        from utils.helpers import bs_delta as _bs_delta

        # Long call: ~0.50–0.55 delta (near ATM)
        long_row = find_strike_by_delta(calls_filtered, 0.50, S, T, r, "call")
        if long_row is None:
            long_row = self._nearest_otm_call(calls_filtered, S, 1.0)

        long_strike = float(long_row["strike"])
        long_iv = float(long_row.get("impliedVolatility", 0.20) or 0.20)
        long_delta_actual = _bs_delta(S, long_strike, T, r, long_iv, "call")

        # Short call: further OTM
        short_delta_target = (delta_targets["short_min"] + delta_targets["short_max"]) / 2
        short_row = find_strike_by_delta(calls_filtered, short_delta_target, S, T, r, "call")
        short_strike = float(short_row["strike"]) if short_row is not None else long_strike + (width_range["min"] + width_range["max"]) / 2
        short_iv = float(short_row.get("impliedVolatility", 0.20) or 0.20) if short_row is not None else long_iv
        short_delta_actual = _bs_delta(S, short_strike, T, r, short_iv, "call")

        long_cost = mid_price(long_row)
        short_credit = self._lookup_strike_mid(calls_filtered, short_strike) or (mid_price(short_row) if short_row is not None else 0.0)
        net_debit = max(long_cost - short_credit, 0.01)
        width = abs(short_strike - long_strike)
        max_risk = net_debit
        breakeven = long_strike + net_debit

        return {
            "short_strike": short_strike,
            "short_delta": short_delta_actual,
            "long_strike": long_strike,
            "long_delta": long_delta_actual,
            "width": width,
            "estimated_credit": -round(net_debit, 2),  # negative = debit
            "max_risk": round(max_risk, 2),
            "breakeven": round(breakeven, 2),
        }

    def _build_bear_put_spread(
        self,
        S: float,
        T: float,
        r: float,
        puts: pd.DataFrame,
        delta_targets: dict,
        width_range: dict,
    ) -> dict:
        """
        Bear Put Spread: buy near-ATM put, sell further OTM put.
        Direction: bearish, debit strategy.
        """
        puts_filtered = filter_chain(
            puts,
            min_open_interest=_OPT_FILTERS["min_open_interest"],
            max_bid_ask_spread_pct=_OPT_FILTERS["max_bid_ask_spread_pct"],
            min_volume=_OPT_FILTERS["min_volume"],
        )
        from utils.helpers import bs_delta as _bs_delta

        # Long put: near ATM (~0.50 delta)
        long_row = find_strike_by_delta(puts_filtered, 0.50, S, T, r, "put")
        if long_row is None:
            long_row = self._nearest_otm_put(puts_filtered, S, 1.0)

        long_strike = float(long_row["strike"])
        long_iv = float(long_row.get("impliedVolatility", 0.20) or 0.20)
        long_delta_actual = _bs_delta(S, long_strike, T, r, long_iv, "put")

        # Short put: further OTM
        short_delta_target = (delta_targets["short_min"] + delta_targets["short_max"]) / 2
        short_row = find_strike_by_delta(puts_filtered, short_delta_target, S, T, r, "put")
        short_strike = float(short_row["strike"]) if short_row is not None else long_strike - (width_range["min"] + width_range["max"]) / 2
        short_iv = float(short_row.get("impliedVolatility", 0.20) or 0.20) if short_row is not None else long_iv
        short_delta_actual = _bs_delta(S, short_strike, T, r, short_iv, "put")

        long_cost = mid_price(long_row)
        short_credit = self._lookup_strike_mid(puts_filtered, short_strike) or (mid_price(short_row) if short_row is not None else 0.0)
        net_debit = max(long_cost - short_credit, 0.01)
        width = abs(long_strike - short_strike)
        max_risk = net_debit
        breakeven = long_strike - net_debit

        return {
            "short_strike": short_strike,
            "short_delta": abs(short_delta_actual),
            "long_strike": long_strike,
            "long_delta": abs(long_delta_actual),
            "width": width,
            "estimated_credit": -round(net_debit, 2),
            "max_risk": round(max_risk, 2),
            "breakeven": round(breakeven, 2),
        }

    def _build_iron_condor(
        self,
        S: float,
        T: float,
        r: float,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        delta_targets: dict,
        width_range: dict,
    ) -> dict:
        """
        Iron Condor: sell OTM put spread + sell OTM call spread.
        Direction: neutral (profit in range-bound market).
        """
        # Build each wing independently
        put_wing = self._build_bull_put_spread(S, T, r, puts, delta_targets, width_range)
        call_wing = self._build_bear_call_spread(S, T, r, calls, delta_targets, width_range)

        total_credit = put_wing["estimated_credit"] + call_wing["estimated_credit"]
        max_wing_risk = max(put_wing["max_risk"], call_wing["max_risk"])
        # IC max risk is max of either wing minus total credit
        max_risk = max_wing_risk - total_credit

        return {
            "short_strike": put_wing["short_strike"],      # put short (lower)
            "short_delta": put_wing["short_delta"],
            "long_strike": call_wing["short_strike"],      # call short (upper)
            "long_delta": call_wing["short_delta"],
            "width": put_wing["width"],                    # wing width
            "estimated_credit": round(total_credit, 2),
            "max_risk": round(max_risk, 2),
            "breakeven": put_wing["breakeven"],            # lower breakeven
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_otm_put(chain: pd.DataFrame, S: float, pct: float = 0.95) -> pd.Series:
        """Return the chain row with strike nearest to S * pct (OTM put proxy)."""
        target = S * pct
        idx = (chain["strike"] - target).abs().idxmin()
        return chain.loc[idx]

    @staticmethod
    def _nearest_otm_call(chain: pd.DataFrame, S: float, pct: float = 1.05) -> pd.Series:
        """Return the chain row with strike nearest to S * pct (OTM call proxy)."""
        target = S * pct
        idx = (chain["strike"] - target).abs().idxmin()
        return chain.loc[idx]

    @staticmethod
    def _lookup_strike_mid(chain: pd.DataFrame, target_strike: float) -> Optional[float]:
        """Look up mid price for a specific strike in a filtered chain."""
        matches = chain[chain["strike"] == target_strike]
        if matches.empty:
            # Find nearest strike within $1
            close = chain[(chain["strike"] - target_strike).abs() <= 1.0]
            if close.empty:
                return None
            matches = close.iloc[[(close["strike"] - target_strike).abs().argmin()]]
        row = matches.iloc[0]
        m = mid_price(row)
        return m if m > 0 else None
