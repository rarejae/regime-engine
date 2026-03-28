"""Backtest report generation — dual-window report with per-source breakdown.

Generates human-readable and machine-readable reports from backtest results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from backtest.runner import BacktestResult, WindowResult


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to stdout.

    Args:
        result: BacktestResult from runner.run_backtest().
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        _print_rich_report(result)
    except ImportError:
        _print_plain_report(result)


def _print_rich_report(result: BacktestResult) -> None:
    """Print report using Rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    console.print(Panel(
        f"Run ID: {result.run_id}\n"
        f"Engine: v{result.engine_version}\n"
        f"Timestamp: {result.run_timestamp}",
        title="[bold]BACKTEST REPORT[/bold]",
        border_style="blue",
    ))

    # Window comparison table
    table = Table(title="Window Comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Pre-COVID", justify="right")
    table.add_column("Post-COVID", justify="right")

    for label, attr in [
        ("Trades", "n_trades"),
        ("Regime Accuracy", "regime_accuracy"),
        ("Strategy Hit Rate", "strategy_hit_rate"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Max Drawdown", "max_drawdown"),
        ("Annualized Return %", "annualized_return_pct"),
    ]:
        pre = getattr(result.pre_covid, attr, "N/A") if result.pre_covid else "N/A"
        post = getattr(result.post_covid, attr, "N/A") if result.post_covid else "N/A"

        if isinstance(pre, float):
            pre = f"{pre:.4f}" if abs(pre) < 1 else f"{pre:.2f}"
        if isinstance(post, float):
            post = f"{post:.4f}" if abs(post) < 1 else f"{post:.2f}"

        table.add_row(label, str(pre), str(post))

    console.print(table)

    # P&L details per window
    for window_result in [result.pre_covid, result.post_covid]:
        if window_result is None:
            continue

        pnl = window_result.realized_pnl
        if not pnl:
            continue

        pnl_table = Table(
            title=f"{window_result.window.replace('_', '-').title()} P&L",
            box=box.SIMPLE,
        )
        pnl_table.add_column("Metric", style="bold")
        pnl_table.add_column("Value", justify="right")

        for key in ["total_pnl", "avg_pnl", "win_rate", "avg_win", "avg_loss",
                     "profit_factor", "max_consecutive_losses", "avg_days_held"]:
            val = pnl.get(key, "N/A")
            if isinstance(val, float):
                if key in ("win_rate",):
                    val = f"{val:.1%}"
                elif key in ("total_pnl", "avg_pnl", "avg_win", "avg_loss"):
                    val = f"${val:.2f}"
                else:
                    val = f"{val:.2f}"
            pnl_table.add_row(key.replace("_", " ").title(), str(val))

        console.print(pnl_table)

        # Strategy breakdown
        by_strategy = pnl.get("by_strategy", {})
        if by_strategy:
            strat_table = Table(
                title=f"{window_result.window.replace('_', '-').title()} by Strategy",
                box=box.SIMPLE,
            )
            strat_table.add_column("Strategy")
            strat_table.add_column("N", justify="right")
            strat_table.add_column("Win Rate", justify="right")
            strat_table.add_column("Total P&L", justify="right")

            for strat, metrics in sorted(by_strategy.items()):
                wr = f"{metrics['win_rate']:.1%}"
                tp = f"${metrics['total_pnl']:.2f}"
                style = "green" if metrics["total_pnl"] > 0 else "red"
                strat_table.add_row(strat, str(metrics["n"]), wr, tp, style=style)

            console.print(strat_table)

    # Per-source breakdown
    if result.by_source:
        source_table = Table(title="Results by Data Source", box=box.SIMPLE)
        source_table.add_column("Source")
        source_table.add_column("Trades", justify="right")
        source_table.add_column("Total P&L", justify="right")
        source_table.add_column("Win Rate", justify="right")

        for src, data in result.by_source.items():
            pnl_data = data.get("realized_pnl", {})
            tp = f"${pnl_data.get('total_pnl', 0):.2f}"
            wr = f"{pnl_data.get('win_rate', 0):.1%}"
            source_table.add_row(src, str(data["n_trades"]), tp, wr)

        console.print(source_table)

    console.print()


def _print_plain_report(result: BacktestResult) -> None:
    """Print plain-text report (fallback)."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT — Run {result.run_id[:8]}")
    print(f"  Engine v{result.engine_version} | {result.run_timestamp}")
    print(f"{'='*60}")

    for window_result in [result.pre_covid, result.post_covid]:
        if window_result is None:
            continue
        print(f"\n--- {window_result.window.upper()} ---")
        print(f"  Period: {window_result.start_date} to {window_result.end_date}")
        print(f"  Trades: {window_result.n_trades}")
        print(f"  Regime Accuracy: {window_result.regime_accuracy:.2%}")
        print(f"  Strategy Hit Rate: {window_result.strategy_hit_rate:.2%}")
        print(f"  Sharpe Ratio: {window_result.sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {window_result.max_drawdown:.2%}")
        print(f"  Annualized Return: {window_result.annualized_return_pct:.2f}%")

        pnl = window_result.realized_pnl
        if pnl:
            print(f"  Total P&L: ${pnl.get('total_pnl', 0):.2f}")
            print(f"  Win Rate: {pnl.get('win_rate', 0):.1%}")

    print()


def save_report_json(result: BacktestResult, output_dir: str = "backtest_results") -> Path:
    """Save the full backtest report as JSON.

    Args:
        result: BacktestResult.
        output_dir: Directory for output files.

    Returns:
        Path to saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"backtest_{result.run_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = out_dir / filename

    # Convert to serializable dict
    data = {
        "run_id": result.run_id,
        "engine_version": result.engine_version,
        "run_timestamp": result.run_timestamp,
        "pre_covid": _window_to_dict(result.pre_covid) if result.pre_covid else None,
        "post_covid": _window_to_dict(result.post_covid) if result.post_covid else None,
        "by_source": result.by_source,
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Backtest report saved to {out_path}")
    return out_path


def _window_to_dict(w: WindowResult) -> dict:
    """Convert WindowResult to a JSON-serializable dict."""
    return {
        "window": w.window,
        "start_date": w.start_date,
        "end_date": w.end_date,
        "n_trades": w.n_trades,
        "regime_accuracy": w.regime_accuracy,
        "strategy_hit_rate": w.strategy_hit_rate,
        "realized_pnl": w.realized_pnl,
        "sharpe_ratio": w.sharpe_ratio,
        "max_drawdown": w.max_drawdown,
        "annualized_return_pct": w.annualized_return_pct,
        "equity_curve": w.equity_curve,
        "trades": [
            {
                "trade_id": t.trade_id,
                "strategy": t.strategy,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
                "win": t.win,
                "days_held": t.days_held,
                "regime_label": t.regime_label,
                "source": t.source,
                "short_strike": t.short_strike,
                "long_strike": t.long_strike,
            }
            for t in w.trades
        ],
    }
