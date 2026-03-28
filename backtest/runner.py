"""Backtest runner — main loop that ties the full pipeline together.

For each trading day in a window:
  1. Load macro snapshot → compute ConditionVector
  2. Score all 5 strategies against ConditionVector
  3. For top strategy: find optimal strikes in that day's option chain
  4. If a valid spread exists and no conflicting open position: open trade
  5. For all open trades: check management rules (take profit, stop loss, regime change, expiry)
  6. Record everything

Runs TWICE: pre-COVID and post-COVID windows, reports results separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from uuid import uuid4

from loguru import logger

from backtest.data_loader import BacktestDataLoader
from backtest.macro_backfill import compute_daily_condition_vectors, build_macro_history
from backtest.metrics import (
    portfolio_metrics,
    realized_pnl_metrics,
    regime_accuracy,
    strategy_hit_rate,
)
from backtest.spread_finder import find_best_spread
from backtest.trade_simulator import (
    BacktestTrade,
    check_management_rules,
    close_trade,
    open_trade,
)
from data.validate_dubach import is_dubach_validated
from engine.condition_vector import ConditionVector
from engine.strategy_scorer import (
    rank_strategies,
    select_dte,
    select_short_delta,
    strategy_type,
)


@dataclass
class WindowResult:
    """Results for a single backtest window."""

    window: str                     # "pre_covid" or "post_covid"
    start_date: str
    end_date: str
    n_trades: int = 0
    regime_accuracy: float = 0.0
    strategy_hit_rate: float = 0.0
    realized_pnl: dict = field(default_factory=dict)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    annualized_return_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Complete backtest results across both windows."""

    run_id: str
    engine_version: str = "2.0"
    run_timestamp: str = ""

    pre_covid: Optional[WindowResult] = None
    post_covid: Optional[WindowResult] = None

    by_source: dict = field(default_factory=dict)


def run_window(
    window: str,
    start_date: date,
    end_date: date,
    condition_vectors: dict[date, ConditionVector],
    loader: BacktestDataLoader,
    run_id: str,
    max_trades: int | None = None,
) -> WindowResult:
    """Run the backtest for a single window.

    Args:
        window: "pre_covid" or "post_covid".
        start_date: Window start.
        end_date: Window end.
        condition_vectors: Pre-computed daily ConditionVectors.
        loader: Data loader for option chains.
        run_id: Run identifier.
        max_trades: Optional cap on total trades (for testing).

    Returns:
        WindowResult with all metrics.
    """
    logger.info(f"Running backtest window: {window} ({start_date} to {end_date})")

    trading_days = loader.trading_days(start_date, end_date)
    logger.info(f"  {len(trading_days)} trading days with data")

    open_trades: list[BacktestTrade] = []
    closed_trades: list[BacktestTrade] = []
    trade_count = 0

    for trade_date in trading_days:
        # Step 1: Get condition vector for this day
        cv = condition_vectors.get(trade_date)
        if cv is None:
            continue

        # Step 2: Score all 5 strategies
        ranked = rank_strategies(cv)
        top_strategy, top_score = ranked[0]

        # Step 3: Open new trade if no open position
        if not open_trades:
            if max_trades is not None and trade_count >= max_trades:
                continue

            target_delta = select_short_delta(cv, top_strategy)
            target_dte = select_dte(cv, strategy_type(top_strategy))

            spread = find_best_spread(
                loader=loader,
                trade_date=trade_date,
                strategy=top_strategy,
                cv=cv,
                target_delta=target_delta,
                target_dte=target_dte,
            )

            if spread is not None and spread.meets_filters():
                trade = open_trade(spread, cv, run_id, window, trade_count)
                open_trades.append(trade)
                trade_count += 1

        # Step 4: Manage open trades
        for trade in open_trades[:]:
            action = check_management_rules(trade, trade_date, cv, loader)
            if action.action == "close":
                trade = close_trade(trade, trade_date, action.reason, loader)
                closed_trades.append(trade)
                open_trades.remove(trade)

    # Force-close any remaining open trades at window end
    for trade in open_trades:
        trade = close_trade(trade, end_date, "window_end", loader)
        closed_trades.append(trade)

    logger.info(f"  Window {window}: {len(closed_trades)} trades closed")

    # Compute metrics
    ra = regime_accuracy(closed_trades)
    shr = strategy_hit_rate(closed_trades, condition_vectors)
    pnl = realized_pnl_metrics(closed_trades)
    portfolio = portfolio_metrics(closed_trades)

    return WindowResult(
        window=window,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        n_trades=len(closed_trades),
        regime_accuracy=ra["accuracy"],
        strategy_hit_rate=shr["hit_rate"],
        realized_pnl=pnl,
        sharpe_ratio=portfolio["sharpe_ratio"],
        max_drawdown=portfolio["max_drawdown"],
        annualized_return_pct=portfolio["annualized_return_pct"],
        equity_curve=portfolio["equity_curve"],
        trades=closed_trades,
    )


def run_backtest(
    macro_start: str = "2005-01-01",
    max_trades_per_window: int | None = None,
) -> BacktestResult:
    """Run the full dual-window backtest.

    Args:
        macro_start: Start date for FRED macro history pull.
        max_trades_per_window: Optional cap on trades per window (for testing).

    Returns:
        BacktestResult with both windows.
    """
    from datetime import datetime

    run_id = str(uuid4())
    logger.info(f"Starting backtest run {run_id}")

    # Determine pre-COVID start based on Dubach validation
    dubach_ok = is_dubach_validated()
    if dubach_ok:
        pre_covid_start = date(2008, 1, 2)
        logger.info("Dubach validated — pre-COVID window starts at 2008")
    else:
        pre_covid_start = date(2010, 1, 4)
        logger.info("Dubach NOT validated — pre-COVID window starts at 2010")

    pre_covid_end = date(2020, 2, 28)
    post_covid_start = date(2020, 3, 1)
    post_covid_end = date(2023, 12, 29)

    # Build macro history and condition vectors
    logger.info("Building macro history from FRED...")
    macro_df = build_macro_history(macro_start)

    logger.info("Computing daily ConditionVectors...")
    condition_vectors = compute_daily_condition_vectors(macro_df)

    # Initialize data loader
    loader = BacktestDataLoader()

    # Run both windows
    pre_covid = run_window(
        "pre_covid", pre_covid_start, pre_covid_end,
        condition_vectors, loader, run_id, max_trades_per_window,
    )

    post_covid = run_window(
        "post_covid", post_covid_start, post_covid_end,
        condition_vectors, loader, run_id, max_trades_per_window,
    )

    # Per-source breakdown
    all_trades = pre_covid.trades + post_covid.trades
    by_source = {}
    sources = set(t.source for t in all_trades)
    for src in sources:
        src_trades = [t for t in all_trades if t.source == src]
        by_source[src] = {
            "n_trades": len(src_trades),
            "realized_pnl": realized_pnl_metrics(src_trades),
            "portfolio": portfolio_metrics(src_trades),
        }

    result = BacktestResult(
        run_id=run_id,
        run_timestamp=datetime.utcnow().isoformat(),
        pre_covid=pre_covid,
        post_covid=post_covid,
        by_source=by_source,
    )

    logger.info(
        f"Backtest complete | "
        f"Pre-COVID: {pre_covid.n_trades} trades, Sharpe={pre_covid.sharpe_ratio:.2f} | "
        f"Post-COVID: {post_covid.n_trades} trades, Sharpe={post_covid.sharpe_ratio:.2f}"
    )

    return result
