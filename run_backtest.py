"""
Macro Regime Engine — Backtest CLI Entry Point

Runs the full dual-window backtest (pre-COVID + post-COVID) and generates
a report with all 4 metric layers.

Usage:
    python run_backtest.py                          # full backtest
    python run_backtest.py --max-trades 10          # limited run for testing
    python run_backtest.py --json                   # save JSON report
    python run_backtest.py --debug                  # verbose logging
"""

from __future__ import annotations

import click
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--max-trades", type=int, default=None, help="Max trades per window (for testing)")
@click.option("--json", "save_json", is_flag=True, help="Save JSON report to backtest_results/")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
@click.option("--macro-start", default="2005-01-01", help="Start date for FRED macro history")
def main(max_trades: int | None, save_json: bool, debug: bool, macro_start: str) -> None:
    """Run the full dual-window backtest and generate report."""

    from utils.logging import setup_logging
    setup_logging("DEBUG" if debug else None)

    from loguru import logger
    logger.info("=" * 60)
    logger.info("MACRO REGIME ENGINE — Backtest Runner")
    logger.info("=" * 60)

    from backtest.runner import run_backtest
    result = run_backtest(
        macro_start=macro_start,
        max_trades_per_window=max_trades,
    )

    from backtest.report import print_report, save_report_json
    print_report(result)

    if save_json:
        path = save_report_json(result)
        click.echo(f"\nJSON report saved to: {path}")


if __name__ == "__main__":
    main()
