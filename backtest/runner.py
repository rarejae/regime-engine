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

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional
from uuid import uuid4

from loguru import logger

from backtest.data_loader import BacktestDataLoader
from backtest.macro_backfill import (
    CONDITION_VECTORS_PATH,
    RAW_INDICATORS_PATH,
    compute_condition_vectors,
    download_raw_indicators,
    load_cached_condition_vectors,
)
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


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
VALIDATION_DIR = Path(__file__).parent.parent / "data" / "validation"


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


# ── Preflight checks ─────────────────────────────────────────────────────────


def preflight_checks(
    macro_start: str = "2005-01-01",
    macro_end: str | None = None,
) -> dict:
    """Validate data availability before starting the backtest.

    Checks:
      1. Options data exists in data/processed/
      2. Raw macro data exists (downloads if missing)
      3. Recomputes condition vectors (always, depends on current weights)
      4. Checks Dubach validation status

    Returns:
        Status dict with data availability info.
    """
    status: dict = {}

    # 1. Check options data
    parquet_files = sorted(PROCESSED_DIR.glob("spy_options_[0-9][0-9][0-9][0-9].parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            "No options data found in data/processed/. "
            "Run: python data/ingest_optionsdx.py"
        )

    options_years = sorted(int(f.stem.replace("spy_options_", "")) for f in parquet_files)
    status["options_years"] = options_years
    logger.info(f"Options data: {min(options_years)}-{max(options_years)} ({len(options_years)} years)")

    # 2. Check raw macro data (download if missing)
    if not RAW_INDICATORS_PATH.exists():
        logger.info("Raw macro data not found. Downloading from FRED (one-time)...")
        download_raw_indicators(macro_start, macro_end)
    else:
        logger.info("Raw macro data: cached (skipping download)")
    status["macro_raw_cached"] = True

    # 3. Always recompute condition vectors (depends on current engine config)
    logger.info("Computing condition vectors with current engine config...")
    compute_condition_vectors()
    status["condition_vectors_computed"] = True

    # 4. Check Dubach validation
    report_path = VALIDATION_DIR / "dubach_validation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        status["dubach_validated"] = report.get("dubach_2008_2009_usable", False)
    else:
        status["dubach_validated"] = False
    status["backtest_start_year"] = 2008 if status["dubach_validated"] else 2010

    # Print summary
    logger.info(f"Options data: {min(options_years)}-{max(options_years)}")
    logger.info(f"Macro data: cached={status['macro_raw_cached']}")
    logger.info(f"Dubach 2008-2009: {'included' if status['dubach_validated'] else 'excluded'}")
    logger.info(f"Backtest window: {status['backtest_start_year']}-{max(options_years)}")

    return status


def print_status() -> None:
    """Print current data availability status without running backtest."""
    print()

    # Options data
    parquet_files = sorted(PROCESSED_DIR.glob("spy_options_[0-9][0-9][0-9][0-9].parquet"))
    if parquet_files:
        years = sorted(int(f.stem.replace("spy_options_", "")) for f in parquet_files)
        total_size_mb = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
        print(f"Options data: {min(years)}-{max(years)} ({len(years)} years, {total_size_mb:.0f} MB)")
    else:
        print("Options data: NOT FOUND — run: python data/ingest_optionsdx.py")

    # Dubach data
    dubach_files = sorted(PROCESSED_DIR.glob("spy_options_*_dubach.parquet"))
    if dubach_files:
        dub_years = sorted(int(f.stem.replace("spy_options_", "").replace("_dubach", "")) for f in dubach_files)
        print(f"Dubach data: {min(dub_years)}-{max(dub_years)} ({len(dub_years)} years)")
    else:
        print("Dubach data: NOT FOUND — run: python data/ingest_dubach.py")

    # Macro data
    if RAW_INDICATORS_PATH.exists():
        import pandas as pd
        raw = pd.read_parquet(RAW_INDICATORS_PATH)
        size_mb = RAW_INDICATORS_PATH.stat().st_size / 1024 / 1024
        print(f"Macro indicators: cached (raw_indicators.parquet, {len(raw):,} rows x {len(raw.columns)} cols, {size_mb:.1f} MB)")
    else:
        print("Macro indicators: NOT CACHED — run: python -m backtest.macro_backfill")

    # Condition vectors
    if CONDITION_VECTORS_PATH.exists():
        print("Condition vectors: cached (will recompute on next backtest run)")
    else:
        print("Condition vectors: not yet computed")

    # Dubach validation
    report_path = VALIDATION_DIR / "dubach_validation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        validated = report.get("dubach_2008_2009_usable", False)
        print(f"Dubach validation: {'PASS' if validated else 'FAIL'}")
    else:
        print("Dubach validation: not run — run: python data/validate_dubach.py")

    # Ready check
    ready = bool(parquet_files) and RAW_INDICATORS_PATH.exists()
    start_year = 2008 if (report_path.exists() and report.get("dubach_2008_2009_usable", False)) else 2010
    if parquet_files:
        print(f"Ready for backtest: {'YES' if ready else 'NO'}")
        if ready:
            print(f"Backtest window: {start_year}-{max(years)}")
    print()


# ── Window runner ─────────────────────────────────────────────────────────────


def run_window(
    window: str,
    start_date: date,
    end_date: date,
    condition_vectors: dict[date, ConditionVector],
    loader: BacktestDataLoader,
    run_id: str,
    max_trades: int | None = None,
) -> WindowResult:
    """Run the backtest for a single window."""
    logger.info(f"Running backtest window: {window} ({start_date} to {end_date})")

    trading_days = loader.trading_days(start_date, end_date)
    logger.info(f"  {len(trading_days)} trading days with data")

    open_trades: list[BacktestTrade] = []
    closed_trades: list[BacktestTrade] = []
    trade_count = 0

    for trade_date in trading_days:
        cv = condition_vectors.get(trade_date)
        if cv is None:
            continue

        ranked = rank_strategies(cv)
        top_strategy, top_score = ranked[0]

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

        for trade in open_trades[:]:
            action = check_management_rules(trade, trade_date, cv, loader)
            if action.action == "close":
                trade = close_trade(trade, trade_date, action.reason, loader)
                closed_trades.append(trade)
                open_trades.remove(trade)

    for trade in open_trades:
        trade = close_trade(trade, end_date, "window_end", loader)
        closed_trades.append(trade)

    logger.info(f"  Window {window}: {len(closed_trades)} trades closed")

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


# ── Main entry point ─────────────────────────────────────────────────────────


def run_backtest(
    macro_start: str = "2005-01-01",
    max_trades_per_window: int | None = None,
    window: str | None = None,
    start_override: date | None = None,
    end_override: date | None = None,
) -> BacktestResult:
    """Run the full dual-window backtest.

    Args:
        macro_start: Start date for FRED macro history pull.
        max_trades_per_window: Optional cap on trades per window.
        window: If set, run only "pre_covid" or "post_covid".
        start_override: Custom start date (overrides window defaults).
        end_override: Custom end date (overrides window defaults).

    Returns:
        BacktestResult with both windows.
    """
    from datetime import datetime

    run_id = str(uuid4())
    logger.info(f"Starting backtest run {run_id}")

    # Preflight checks (downloads macro if needed, always recomputes CVs)
    status = preflight_checks(macro_start)

    # Load condition vectors
    condition_vectors = load_cached_condition_vectors()
    if condition_vectors is None:
        raise RuntimeError("Condition vectors not computed — preflight should have done this")

    # Initialize data loader
    loader = BacktestDataLoader()

    # Determine window boundaries
    dubach_ok = status["dubach_validated"]
    pre_covid_start = date(2008, 1, 2) if dubach_ok else date(2010, 1, 4)
    pre_covid_end = date(2020, 2, 28)
    post_covid_start = date(2020, 3, 1)
    post_covid_end = date(2023, 12, 29)

    # Custom date range overrides both windows into a single window
    if start_override and end_override:
        result_window = run_window(
            "custom", start_override, end_override,
            condition_vectors, loader, run_id, max_trades_per_window,
        )
        return BacktestResult(
            run_id=run_id,
            run_timestamp=datetime.utcnow().isoformat(),
            pre_covid=result_window,
            post_covid=None,
        )

    pre_covid = None
    post_covid = None

    if window is None or window == "pre_covid":
        pre_covid = run_window(
            "pre_covid", pre_covid_start, pre_covid_end,
            condition_vectors, loader, run_id, max_trades_per_window,
        )

    if window is None or window == "post_covid":
        post_covid = run_window(
            "post_covid", post_covid_start, post_covid_end,
            condition_vectors, loader, run_id, max_trades_per_window,
        )

    # Per-source breakdown
    all_trades = (pre_covid.trades if pre_covid else []) + (post_covid.trades if post_covid else [])
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

    if pre_covid:
        logger.info(f"Pre-COVID: {pre_covid.n_trades} trades, Sharpe={pre_covid.sharpe_ratio:.2f}")
    if post_covid:
        logger.info(f"Post-COVID: {post_covid.n_trades} trades, Sharpe={post_covid.sharpe_ratio:.2f}")

    return result
