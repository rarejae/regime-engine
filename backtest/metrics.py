"""Backtest metrics — all 4 metric layers.

Layer 1: Regime Classification Accuracy
Layer 2: Strategy Selection Hit Rate
Layer 3: Realized P&L with Management Rules
Layer 4: Sharpe Ratio and Max Drawdown
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import mean
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.trade_simulator import BacktestTrade
from engine.condition_vector import ConditionVector
from engine.strategy_scorer import rank_strategies


# ── Layer 1: Regime Classification Accuracy ───────────────────────────────────

def regime_accuracy(trades: list[BacktestTrade], spy_returns: dict[date, float] | None = None) -> dict:
    """Did the regime engine correctly classify what actually happened?

    Compares regime-implied direction at entry to actual SPY move over the trade's life.

    Args:
        trades: List of closed backtest trades.
        spy_returns: Optional dict mapping date → cumulative return since trade open.

    Returns:
        Dict with accuracy and trade count.
    """
    if not trades:
        return {"accuracy": 0.0, "n": 0}

    correct = 0
    evaluated = 0

    for t in trades:
        if t.underlying_at_entry is None or t.underlying_at_exit is None:
            continue
        if t.underlying_at_entry == 0:
            continue

        actual_return = (t.underlying_at_exit - t.underlying_at_entry) / t.underlying_at_entry

        # Direction implied by strategy
        if t.strategy in ("bull_put_spread", "bull_call_spread"):
            predicted_direction = "bullish"
        elif t.strategy in ("bear_call_spread", "bear_put_spread"):
            predicted_direction = "bearish"
        else:
            predicted_direction = "neutral"

        actual_direction = "bullish" if actual_return > 0.005 else ("bearish" if actual_return < -0.005 else "neutral")

        if predicted_direction == actual_direction:
            correct += 1
        elif predicted_direction == "neutral" and abs(actual_return) < 0.03:
            correct += 1  # neutral is correct if market didn't move much

        evaluated += 1

    accuracy = correct / evaluated if evaluated > 0 else 0.0
    return {"accuracy": round(accuracy, 4), "n": evaluated}


# ── Layer 2: Strategy Selection Hit Rate ──────────────────────────────────────

def strategy_hit_rate(
    trades: list[BacktestTrade],
    condition_vectors: dict[date, ConditionVector] | None = None,
) -> dict:
    """Did the top-scoring strategy outperform the alternatives?

    Args:
        trades: List of closed trades.
        condition_vectors: Dict mapping date → ConditionVector.

    Returns:
        Dict with hit_rate and count.
    """
    if not trades or condition_vectors is None:
        return {"hit_rate": 0.0, "n": 0}

    top_was_best = 0
    evaluated = 0

    for t in trades:
        if t.entry_date not in condition_vectors:
            continue

        cv = condition_vectors[t.entry_date]
        ranked = rank_strategies(cv)

        # The top strategy is what was actually traded
        top_strategy = ranked[0][0]
        if top_strategy == t.strategy and t.win:
            top_was_best += 1
        evaluated += 1

    hit_rate = top_was_best / evaluated if evaluated > 0 else 0.0
    return {"hit_rate": round(hit_rate, 4), "n": evaluated}


# ── Layer 3: Realized P&L ─────────────────────────────────────────────────────

def realized_pnl_metrics(trades: list[BacktestTrade]) -> dict:
    """Compute comprehensive P&L metrics from trade list.

    Args:
        trades: List of closed trades.

    Returns:
        Dict with P&L statistics, grouped by strategy, regime, and source.
    """
    if not trades:
        return _empty_pnl_metrics()

    pnls = [t.pnl for t in trades]
    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]

    # Profit factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1  # avoid div/0

    # Max consecutive losses
    max_consec = _max_consecutive(trades, lambda t: not t.win)

    result = {
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(mean(pnls), 4),
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 4),
        "avg_win": round(mean([t.pnl for t in wins]), 4) if wins else 0.0,
        "avg_loss": round(mean([t.pnl for t in losses]), 4) if losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "max_consecutive_losses": max_consec,
        "avg_days_held": round(mean([t.days_held for t in trades]), 1),
        "by_strategy": _group_metrics(trades, lambda t: t.strategy),
        "by_regime": _group_metrics(trades, lambda t: t.regime_label),
        "by_source": _group_metrics(trades, lambda t: t.source),
        "by_exit_reason": _group_metrics(trades, lambda t: t.exit_reason or "unknown"),
    }

    return result


def _empty_pnl_metrics() -> dict:
    return {
        "total_pnl": 0.0, "avg_pnl": 0.0, "n_trades": 0,
        "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "max_consecutive_losses": 0,
        "avg_days_held": 0.0,
        "by_strategy": {}, "by_regime": {}, "by_source": {}, "by_exit_reason": {},
    }


def _group_metrics(trades: list[BacktestTrade], key_fn) -> dict:
    """Group trades by a key function and compute metrics per group."""
    groups: dict[str, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)

    result = {}
    for group_key, group_trades in groups.items():
        wins = [t for t in group_trades if t.win]
        pnls = [t.pnl for t in group_trades]
        result[group_key] = {
            "n": len(group_trades),
            "total_pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / len(group_trades), 4) if group_trades else 0.0,
            "avg_pnl": round(mean(pnls), 4) if pnls else 0.0,
        }
    return result


def _max_consecutive(trades: list[BacktestTrade], condition) -> int:
    """Count the maximum consecutive trades matching a condition."""
    max_count = 0
    current = 0
    for t in sorted(trades, key=lambda x: x.entry_date):
        if condition(t):
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


# ── Layer 4: Portfolio Metrics (Sharpe, Drawdown) ─────────────────────────────

def portfolio_metrics(
    trades: list[BacktestTrade],
    initial_capital: float = 100_000.0,
) -> dict:
    """Compute equity curve, Sharpe ratio, and max drawdown.

    Args:
        trades: List of closed trades (sorted by exit_date internally).
        initial_capital: Starting portfolio value.

    Returns:
        Dict with Sharpe, max drawdown, total return, equity curve.
    """
    if not trades:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": 0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "equity_curve": [initial_capital],
        }

    # Sort by exit date
    sorted_trades = sorted(trades, key=lambda x: x.exit_date or x.entry_date)

    # Build equity curve (P&L per share × 100 shares per contract)
    equity = [initial_capital]
    for t in sorted_trades:
        contract_pnl = t.pnl * 100  # per-contract P&L
        equity.append(equity[-1] + contract_pnl)

    equity_series = pd.Series(equity)

    # Returns
    returns = equity_series.pct_change().dropna()

    # Sharpe ratio (annualized)
    if len(returns) > 1 and returns.std() > 0:
        avg_holding = mean([t.days_held for t in sorted_trades]) or 30
        trades_per_year = 252 / avg_holding
        sharpe = float(returns.mean() / returns.std() * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = float(drawdown.min())

    # Max drawdown duration
    dd_duration = _max_drawdown_duration(equity)

    # Total and annualized return
    total_return_pct = (equity[-1] - initial_capital) / initial_capital * 100

    # Annualized return
    if sorted_trades:
        first_date = sorted_trades[0].entry_date
        last_date = sorted_trades[-1].exit_date or sorted_trades[-1].entry_date
        total_days = (last_date - first_date).days
        if total_days > 0:
            years = total_days / 365.25
            annualized = ((equity[-1] / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized = 0.0
    else:
        annualized = 0.0

    return {
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_duration_days": dd_duration,
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized, 2),
        "equity_curve": [round(e, 2) for e in equity],
    }


def _max_drawdown_duration(equity: list[float]) -> int:
    """Compute the longest period (in trades) spent in drawdown."""
    peak = equity[0]
    in_dd_since = 0
    max_duration = 0
    current_duration = 0

    for i, val in enumerate(equity):
        if val >= peak:
            peak = val
            if current_duration > max_duration:
                max_duration = current_duration
            current_duration = 0
        else:
            current_duration += 1

    return max(max_duration, current_duration)
