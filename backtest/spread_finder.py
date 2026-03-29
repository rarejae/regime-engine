"""Spread finder — locates optimal strikes in historical option chains.

Given a ConditionVector, strategy, and target parameters, finds the best
spread in the available option chain for a given trading day.

Uses DuckDB queries against Parquet files via BacktestDataLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.data_loader import BacktestDataLoader
from engine.condition_vector import ConditionVector
from engine.strategy_scorer import strategy_type, STRATEGY_DISPLAY
from utils.helpers import bs_delta


@dataclass
class SpreadCandidate:
    """A candidate vertical spread found in the historical chain."""

    strategy: str
    trade_date: date
    expiry: date
    dte: int
    short_strike: float
    long_strike: float
    short_delta: float
    long_delta: float
    short_mid: float
    long_mid: float
    width: float
    entry_credit: float        # positive for credit, negative for debit
    max_risk: float
    breakeven: float
    option_type: str           # 'C' or 'P'
    underlying_close: float
    source: str
    short_iv: float
    long_iv: float

    def meets_filters(
        self,
        min_credit_to_width: float = 0.15,
        max_risk: float = 500.0,
        min_width: float = 1.0,
    ) -> bool:
        """Check if this spread meets minimum quality filters."""
        if self.width < min_width:
            return False
        if self.max_risk > max_risk:
            return False
        if self.entry_credit > 0:  # credit spread
            credit_to_width = self.entry_credit / self.width if self.width > 0 else 0
            if credit_to_width < min_credit_to_width:
                return False
        return True


def find_best_spread(
    loader: BacktestDataLoader,
    trade_date: date,
    strategy: str,
    cv: ConditionVector,
    target_delta: float,
    target_dte: int,
) -> Optional[SpreadCandidate]:
    """Find the best spread in the historical chain for a given day.

    Args:
        loader: BacktestDataLoader for querying Parquet files.
        trade_date: Trading day.
        strategy: Strategy key (e.g. "bear_call_spread").
        cv: ConditionVector for context.
        target_delta: Target short delta magnitude.
        target_dte: Target DTE.

    Returns:
        Best SpreadCandidate, or None if no valid spread found.
    """
    strat_type = strategy_type(strategy)

    # Determine option type for legs
    if strategy in ("bull_put_spread", "bear_put_spread"):
        opt_type = "P"
    elif strategy in ("bear_call_spread", "bull_call_spread"):
        opt_type = "C"
    elif strategy == "iron_condor":
        # For iron condor, find put wing and call wing separately
        put_spread = _find_wing(loader, trade_date, "P", target_delta, target_dte, cv, "bull_put_spread")
        call_spread = _find_wing(loader, trade_date, "C", target_delta, target_dte, cv, "bear_call_spread")
        if put_spread is None or call_spread is None:
            return None
        # Combine into single spread representation (use put wing as primary)
        total_credit = put_spread.entry_credit + call_spread.entry_credit
        return SpreadCandidate(
            strategy="iron_condor",
            trade_date=trade_date,
            expiry=put_spread.expiry,
            dte=put_spread.dte,
            short_strike=put_spread.short_strike,
            long_strike=call_spread.short_strike,
            short_delta=put_spread.short_delta,
            long_delta=call_spread.short_delta,
            short_mid=put_spread.short_mid + call_spread.short_mid,
            long_mid=put_spread.long_mid + call_spread.long_mid,
            width=put_spread.width,
            entry_credit=total_credit,
            max_risk=max(put_spread.width, call_spread.width) - total_credit,
            breakeven=put_spread.breakeven,
            option_type="IC",
            underlying_close=put_spread.underlying_close,
            source=put_spread.source,
            short_iv=put_spread.short_iv,
            long_iv=call_spread.short_iv,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return _find_wing(loader, trade_date, opt_type, target_delta, target_dte, cv, strategy)


def _find_wing(
    loader: BacktestDataLoader,
    trade_date: date,
    opt_type: str,
    target_delta: float,
    target_dte: int,
    cv: ConditionVector,
    strategy: str,
) -> Optional[SpreadCandidate]:
    """Find the best single-wing spread (put or call side).

    Args:
        loader: Data loader.
        trade_date: Trading day.
        opt_type: 'C' or 'P'.
        target_delta: Target delta magnitude for short leg.
        target_dte: Target DTE.
        cv: ConditionVector.
        strategy: Strategy name for context.

    Returns:
        SpreadCandidate or None.
    """
    # Query chain with some DTE flexibility
    dte_min = max(7, target_dte - 14)
    dte_max = target_dte + 14

    chain = loader.get_chain(
        trade_date=trade_date,
        dte_range=(dte_min, dte_max),
        option_type=opt_type,
        min_oi=0,  # optionsDX free EOD data does not include open_interest
    )

    if chain.empty:
        logger.debug(f"No chain data for {trade_date} {opt_type}")
        return None

    # Filter out rows with no market (bid=0 and ask=0)
    chain = chain[(chain["bid"] > 0) | (chain["ask"] > 0)].copy()
    if chain.empty:
        logger.debug(f"No priced contracts for {trade_date} {opt_type}")
        return None

    # Get underlying close
    underlying_close = chain["underlying_close"].iloc[0] if "underlying_close" in chain.columns else 0.0

    # Get risk-free rate approximation
    r = 0.04  # reasonable approximation for backtest

    # Find the expiry closest to target DTE
    chain["dte_diff"] = abs(chain["dte"] - target_dte)
    best_dte = chain.loc[chain["dte_diff"].idxmin(), "dte"]
    best_expiry = chain.loc[chain["dte_diff"].idxmin(), "expiry"]
    chain_exp = chain[chain["expiry"] == best_expiry].copy()

    if chain_exp.empty:
        return None

    # Compute delta if not reliable in data
    T = best_dte / 365.0
    chain_exp = chain_exp.copy()
    computed_deltas = []
    for _, row in chain_exp.iterrows():
        iv = row.get("iv", 0)
        if iv is None or iv <= 0 or np.isnan(iv):
            iv = 0.20
        try:
            d = bs_delta(underlying_close, float(row["strike"]), T, r, float(iv), opt_type.lower())
            computed_deltas.append(abs(d))
        except Exception:
            computed_deltas.append(np.nan)
    chain_exp["abs_delta"] = computed_deltas
    chain_exp = chain_exp.dropna(subset=["abs_delta"])

    if chain_exp.empty:
        return None

    # Find short strike closest to target delta
    chain_exp["delta_diff"] = abs(chain_exp["abs_delta"] - target_delta)
    short_idx = chain_exp["delta_diff"].idxmin()
    short_row = chain_exp.loc[short_idx]

    short_strike = float(short_row["strike"])
    short_mid = float(short_row["mid"]) if short_row["mid"] > 0 else (float(short_row["bid"]) + float(short_row["ask"])) / 2
    short_delta_actual = float(short_row["abs_delta"])
    short_iv = float(short_row.get("iv", 0.20) or 0.20)
    source = str(short_row.get("source", "unknown"))

    # Determine spread width based on VIX level
    if cv.vix < 15:
        target_width = 3.0
    elif cv.vix < 25:
        target_width = 5.0
    elif cv.vix < 40:
        target_width = 7.0
    else:
        target_width = 10.0

    # Find long strike (further OTM by target_width)
    if strategy in ("bull_put_spread",):
        # Long put is lower strike
        long_target = short_strike - target_width
    elif strategy in ("bear_call_spread",):
        # Long call is higher strike
        long_target = short_strike + target_width
    elif strategy in ("bull_call_spread",):
        # Short call is higher, long call is lower (closer to ATM)
        long_target = short_strike - target_width
    elif strategy in ("bear_put_spread",):
        # Short put is lower, long put is higher (closer to ATM)
        long_target = short_strike + target_width
    else:
        long_target = short_strike + target_width

    # Find closest available strike to long target
    chain_exp["long_diff"] = abs(chain_exp["strike"] - long_target)
    long_candidates = chain_exp[chain_exp["strike"] != short_strike]
    if long_candidates.empty:
        return None

    long_idx = long_candidates["long_diff"].idxmin()
    long_row = chain_exp.loc[long_idx]

    long_strike = float(long_row["strike"])
    long_mid = float(long_row["mid"]) if long_row["mid"] > 0 else (float(long_row["bid"]) + float(long_row["ask"])) / 2
    long_delta_actual = float(long_row["abs_delta"])
    long_iv = float(long_row.get("iv", 0.20) or 0.20)

    width = abs(short_strike - long_strike)
    if width == 0:
        return None

    # Compute entry credit/debit
    strat_type = strategy_type(strategy)
    if strat_type == "credit":
        entry_credit = short_mid - long_mid
        max_risk = width - max(entry_credit, 0)
        if strategy in ("bull_put_spread",):
            breakeven = short_strike - max(entry_credit, 0)
        else:
            breakeven = short_strike + max(entry_credit, 0)
    else:
        # Debit: pay long_mid, receive short_mid
        entry_debit = long_mid - short_mid
        entry_credit = -entry_debit
        max_risk = entry_debit
        if strategy in ("bull_call_spread",):
            breakeven = long_strike + entry_debit
        else:
            breakeven = long_strike - entry_debit

    expiry_date = pd.Timestamp(best_expiry).date() if hasattr(best_expiry, 'date') else best_expiry

    return SpreadCandidate(
        strategy=strategy,
        trade_date=trade_date,
        expiry=expiry_date,
        dte=int(best_dte),
        short_strike=short_strike,
        long_strike=long_strike,
        short_delta=short_delta_actual,
        long_delta=long_delta_actual,
        short_mid=short_mid,
        long_mid=long_mid,
        width=width,
        entry_credit=round(entry_credit, 4),
        max_risk=round(max_risk, 4),
        breakeven=round(breakeven, 2),
        option_type=opt_type,
        underlying_close=underlying_close,
        source=source,
        short_iv=short_iv,
        long_iv=long_iv,
    )
