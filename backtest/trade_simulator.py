"""Trade simulator — manages entries, tracking, and exits with management rules.

Simulates the lifecycle of a spread trade: entry at observed prices,
daily mark-to-market from Parquet data, and exit when management rules trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from uuid import UUID

from loguru import logger

from backtest.data_loader import BacktestDataLoader
from backtest.spread_finder import SpreadCandidate
from engine.condition_vector import ConditionVector
from engine.strategy_scorer import (
    opposite_strategy,
    score_strategy,
    strategy_type,
    STRATEGY_DISPLAY,
)


@dataclass
class BacktestTrade:
    """A single backtest trade record."""

    trade_id: str
    run_id: str
    window: str                   # "pre_covid" or "post_covid"

    # Entry
    strategy: str
    strategy_display: str
    strategy_type: str            # "credit" or "debit"
    entry_date: date
    expiry: date
    dte_at_entry: int
    short_strike: float
    long_strike: float
    short_delta: float
    long_delta: float
    width: float
    entry_credit: float           # positive for credit, negative for debit
    option_type: str
    underlying_at_entry: float
    source: str                   # data source tag

    # Condition vector at entry
    regime_label: str
    conviction: float
    growth: float
    inflation: float
    liquidity: float
    risk: float

    # Exit
    exit_date: Optional[date] = None
    exit_value: Optional[float] = None
    exit_reason: Optional[str] = None
    underlying_at_exit: Optional[float] = None
    pnl: float = 0.0
    win: bool = False

    # Tracking
    max_favorable: float = 0.0    # best P&L seen during trade
    max_adverse: float = 0.0      # worst P&L seen during trade
    days_held: int = 0


def open_trade(
    spread: SpreadCandidate,
    cv: ConditionVector,
    run_id: str,
    window: str,
    trade_count: int,
) -> BacktestTrade:
    """Create a BacktestTrade from a SpreadCandidate.

    Args:
        spread: The spread to open.
        cv: ConditionVector at time of entry.
        run_id: Backtest run identifier.
        window: "pre_covid" or "post_covid".
        trade_count: Running trade counter for ID generation.

    Returns:
        New BacktestTrade in open state.
    """
    trade = BacktestTrade(
        trade_id=f"{run_id[:8]}-{trade_count:04d}",
        run_id=run_id,
        window=window,
        strategy=spread.strategy,
        strategy_display=STRATEGY_DISPLAY.get(spread.strategy, spread.strategy),
        strategy_type=strategy_type(spread.strategy),
        entry_date=spread.trade_date,
        expiry=spread.expiry,
        dte_at_entry=spread.dte,
        short_strike=spread.short_strike,
        long_strike=spread.long_strike,
        short_delta=spread.short_delta,
        long_delta=spread.long_delta,
        width=spread.width,
        entry_credit=spread.entry_credit,
        option_type=spread.option_type,
        underlying_at_entry=spread.underlying_close,
        source=spread.source,
        regime_label=cv.regime_label,
        conviction=cv.conviction,
        growth=cv.growth,
        inflation=cv.inflation,
        liquidity=cv.liquidity,
        risk=cv.risk,
    )

    logger.debug(
        f"Opened trade {trade.trade_id}: {trade.strategy_display} "
        f"{trade.short_strike}/{trade.long_strike} "
        f"exp={trade.expiry} credit={trade.entry_credit:.2f}"
    )
    return trade


@dataclass
class Action:
    """Result of checking management rules."""
    action: str   # "hold" or "close"
    reason: str = ""


def check_management_rules(
    trade: BacktestTrade,
    current_date: date,
    cv: ConditionVector,
    loader: BacktestDataLoader,
) -> Action:
    """Check if a trade should be closed based on management rules.

    Args:
        trade: The open trade.
        current_date: Current backtest date.
        cv: Current ConditionVector.
        loader: Data loader for current spread pricing.

    Returns:
        Action indicating whether to hold or close (with reason).
    """
    # Days to expiry
    dte_remaining = (trade.expiry - current_date).days

    # Close at 7 DTE to avoid gamma risk
    if dte_remaining <= 7:
        return Action("close", "approaching_expiry")

    # Get current spread value
    current_value = loader.get_spread_value(
        trade_date=current_date,
        short_strike=trade.short_strike,
        long_strike=trade.long_strike,
        expiry=trade.expiry,
        option_type=trade.option_type if trade.option_type != "IC" else "P",
        source=None,
    )

    if current_value is not None and trade.entry_credit > 0:
        # CREDIT spread: get_spread_value returns short_mid - long_mid (positive).
        # This is the "cost to close" the position.
        # P&L = entry_credit - cost_to_close.  Profit when cost_to_close shrinks.
        cost_to_close = current_value
        pnl = trade.entry_credit - cost_to_close

        # Take profit at 50% of max profit
        if pnl >= trade.entry_credit * 0.50:
            return Action("close", "take_profit_50pct")

        # Stop loss: lost more than the credit received (cost doubled)
        if cost_to_close >= trade.entry_credit * 2:
            return Action("close", "stop_loss_2x")

    elif current_value is not None and trade.entry_credit < 0:
        # DEBIT spread: get_spread_value returns short_mid - long_mid (NEGATIVE,
        # because the long leg is worth more than the short leg).
        # The spread's value to us = -current_value = long_mid - short_mid (positive).
        entry_debit = abs(trade.entry_credit)
        spread_value_to_us = abs(current_value)

        # Take profit: spread now worth 2x what we paid
        if spread_value_to_us >= entry_debit * 2:
            return Action("close", "take_profit_100pct")

        # Stop loss: spread lost 50% of its value
        if spread_value_to_us <= entry_debit * 0.50:
            return Action("close", "stop_loss_50pct")

    # Regime reversal check
    opp = opposite_strategy(trade.strategy)
    if opp != trade.strategy:
        opposite_score = score_strategy(opp, cv)
        current_score = score_strategy(trade.strategy, cv)
        if opposite_score > current_score + 30:
            return Action("close", "regime_reversal")

    return Action("hold")


def close_trade(
    trade: BacktestTrade,
    current_date: date,
    reason: str,
    loader: BacktestDataLoader,
) -> BacktestTrade:
    """Close a trade and compute final P&L.

    Args:
        trade: The trade to close.
        current_date: Date of closure.
        reason: Reason for closing.
        loader: Data loader for exit pricing.

    Returns:
        Updated trade with exit fields filled.
    """
    # Get exit spread value
    exit_value = loader.get_spread_value(
        trade_date=current_date,
        short_strike=trade.short_strike,
        long_strike=trade.long_strike,
        expiry=trade.expiry,
        option_type=trade.option_type if trade.option_type != "IC" else "P",
    )

    # Get underlying at exit
    underlying_at_exit = loader.get_spy_close(current_date)

    if exit_value is None:
        # If no data available, estimate based on reason
        if reason == "approaching_expiry":
            # Near expiry: estimate from intrinsic value
            if underlying_at_exit is not None:
                if trade.option_type == "P":
                    short_intrinsic = max(0, trade.short_strike - underlying_at_exit)
                    long_intrinsic = max(0, trade.long_strike - underlying_at_exit)
                else:
                    short_intrinsic = max(0, underlying_at_exit - trade.short_strike)
                    long_intrinsic = max(0, underlying_at_exit - trade.long_strike)
                exit_value = short_intrinsic - long_intrinsic
            else:
                exit_value = 0.0
        else:
            exit_value = trade.entry_credit * 0.5  # conservative estimate

    # Compute P&L (per share).
    # exit_value = short_mid - long_mid from get_spread_value.
    # For credit spreads: P&L = entry_credit - exit_value (profit when exit_value shrinks)
    # For debit spreads: P&L = abs(exit_value) - abs(entry_credit) (profit when spread widens)
    if trade.strategy_type == "credit":
        pnl = trade.entry_credit - exit_value
    else:
        pnl = abs(exit_value) - abs(trade.entry_credit)

    # Apply realistic slippage: crossing half the bid-ask spread on each leg, entry + exit.
    # Iron condors have 4 legs; all other spreads have 2 legs.
    # Conservative estimates based on SPY options typical bid-ask spreads.
    if trade.strategy == "iron_condor":
        slippage = 0.30  # 4 legs × ~$0.075 per half-spread crossing
    else:
        slippage = 0.08  # 2 legs × ~$0.04 per half-spread crossing
    pnl -= slippage

    trade.exit_date = current_date
    trade.exit_value = exit_value
    trade.exit_reason = reason
    trade.underlying_at_exit = underlying_at_exit
    trade.pnl = round(pnl, 4)
    trade.win = pnl > 0
    trade.days_held = (current_date - trade.entry_date).days

    logger.debug(
        f"Closed trade {trade.trade_id}: {reason} | "
        f"P&L=${trade.pnl:.2f} | held={trade.days_held}d | "
        f"{'WIN' if trade.win else 'LOSS'}"
    )
    return trade
