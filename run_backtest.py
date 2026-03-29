"""
Macro Regime Engine — Backtest CLI Entry Point

Runs the full dual-window backtest (pre-COVID + post-COVID) and generates
a report with all 4 metric layers.

Usage:
    python run_backtest.py                                  # full backtest (both windows)
    python run_backtest.py --window pre_covid                # single window
    python run_backtest.py --window post_covid               # single window
    python run_backtest.py --start 2020-01-01 --end 2020-12-31  # custom date range
    python run_backtest.py --max-trades 10                   # limited run for testing
    python run_backtest.py --json                            # save JSON report
    python run_backtest.py --status                          # show data availability
    python run_backtest.py --debug                           # verbose logging
"""

from __future__ import annotations

from datetime import date

import click
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--max-trades", type=int, default=None, help="Max trades per window (for testing)")
@click.option("--json", "save_json", is_flag=True, help="Save JSON report to backtest_results/")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
@click.option("--macro-start", default="2005-01-01", help="Start date for FRED macro history")
@click.option("--window", type=click.Choice(["pre_covid", "post_covid"]), default=None, help="Run only one window")
@click.option("--start", "start_date", type=str, default=None, help="Custom start date (YYYY-MM-DD)")
@click.option("--end", "end_date", type=str, default=None, help="Custom end date (YYYY-MM-DD)")
@click.option("--status", is_flag=True, help="Show data availability and exit")
def main(
    max_trades: int | None,
    save_json: bool,
    debug: bool,
    macro_start: str,
    window: str | None,
    start_date: str | None,
    end_date: str | None,
    status: bool,
) -> None:
    """Run the full dual-window backtest and generate report."""

    from utils.logging import setup_logging
    setup_logging("DEBUG" if debug else None)

    # --status: just print data availability and exit
    if status:
        from backtest.runner import print_status
        print_status()
        return

    from loguru import logger
    logger.info("=" * 60)
    logger.info("MACRO REGIME ENGINE — Backtest Runner")
    logger.info("=" * 60)

    # Parse custom date range
    start_override = None
    end_override = None
    if start_date and end_date:
        start_override = date.fromisoformat(start_date)
        end_override = date.fromisoformat(end_date)
    elif start_date or end_date:
        raise click.UsageError("Both --start and --end must be provided together")

    from backtest.runner import run_backtest
    result = run_backtest(
        macro_start=macro_start,
        max_trades_per_window=max_trades,
        window=window,
        start_override=start_override,
        end_override=end_override,
    )

    from backtest.report import print_report, save_report_json
    print_report(result)

    if save_json:
        path = save_report_json(result)
        click.echo(f"\nJSON report saved to: {path}")


if __name__ == "__main__":
    main()
