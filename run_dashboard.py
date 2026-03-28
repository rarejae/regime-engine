"""
Macro Regime Engine — Live Terminal Dashboard

Displays a continuously updating regime dashboard using the rich library.
Refreshes on a configurable interval (default: 15 minutes for real-time data).

Usage:
    python run_dashboard.py
    python run_dashboard.py --interval 5   # refresh every 5 minutes
    python run_dashboard.py --once         # single snapshot then exit
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import click
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--interval", default=15, help="Refresh interval in minutes (default: 15)")
@click.option("--once", is_flag=True, help="Run once then exit (no loop)")
def run_dashboard(interval: int, once: bool) -> None:
    """Live terminal dashboard for the Macro Regime Engine."""

    from utils.logging import setup_logging
    setup_logging("WARNING")  # quiet logging for dashboard mode

    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    console = Console()

    def build_display(card, snapshot, elapsed: float) -> Layout:
        """Assemble the full Rich Layout from a completed TradeCard."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        # Header
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        header_text = Text(f"  MACRO REGIME ENGINE  ·  {now}  ·  next refresh in {int(interval*60 - elapsed)}s", style="bold white on dark_blue")
        layout["header"].update(Panel(header_text, box=box.HORIZONTALS))

        # Left panel: Regime + Axis Scores
        regime_color = {"GOLDILOCKS": "green", "REFLATION": "yellow", "STAGFLATION": "red", "DEFLATION": "bright_red", "MIXED": "cyan"}.get(card.regime, "white")
        axis_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        axis_table.add_column("Axis", style="bold", min_width=12)
        axis_table.add_column("Score", justify="right", min_width=8)
        axis_table.add_column("Bar", min_width=15)
        for axis, score in card.axis_scores.items():
            bar_len = int(abs(score) / 10)
            bar = ("▇" * bar_len) if bar_len > 0 else "·"
            color = "green" if score >= 0 else "red"
            axis_table.add_row(
                axis,
                f"[{color}]{score:+.1f}[/{color}]",
                f"[{color}]{bar}[/{color}]",
            )
        regime_section = (
            f"[bold {regime_color}]{card.regime}[/bold {regime_color}]\n"
            f"Conviction: [bold]{card.conviction:.0f}[/bold]/100  ·  {card.transition}\n"
            + (f"Modifiers: [yellow]{', '.join(card.modifiers)}[/yellow]" if card.modifiers else "Modifiers: [dim]none[/dim]")
        )
        layout["left"].update(Panel(
            regime_section + "\n\n" + _table_to_str(axis_table),
            title="[bold]REGIME[/bold]",
            border_style=regime_color,
        ))

        # Right panel: Trade Card
        is_credit = card.strategy_type == "credit"
        credit_label = "Net Credit" if is_credit else "Net Debit"
        trade_table = Table(box=box.SIMPLE, show_header=False, min_width=40)
        trade_table.add_column("Field", style="bold", min_width=16)
        trade_table.add_column("Value")
        trade_table.add_row("Strategy", f"[bold]{card.strategy}[/bold]")
        trade_table.add_row("Expiry / DTE", f"{card.expiry}  ({card.dte}d)")
        trade_table.add_row("Short Strike", f"${card.short_strike:.0f}  (Δ {card.short_delta:.3f})")
        trade_table.add_row("Long Strike", f"${card.long_strike:.0f}  (Δ {card.long_delta:.3f})")
        trade_table.add_row("Width", f"${card.width:.0f}")
        trade_table.add_row(credit_label, f"[bold]${abs(card.estimated_credit):.2f}[/bold]  (${abs(card.estimated_credit)*100:.0f}/contract)")
        trade_table.add_row("Max Risk", f"${card.max_risk:.2f}/share")
        trade_table.add_row("Breakeven", f"${card.breakeven:.2f}")
        trade_table.add_row("Credit/Width", f"{card.credit_to_width:.1%}")
        win_color = "green" if card.win_probability > 0.60 else "yellow"
        trade_table.add_row("Win Prob", f"[bold {win_color}]{card.win_probability*100:.1f}%[/bold {win_color}]")

        layout["right"].update(Panel(
            trade_table,
            title="[bold]TRADE CARD[/bold]",
            border_style="green",
        ))

        # Footer: notes
        notes = " · ".join(card.notes[:3]) if card.notes else "No proxy indicators"
        layout["footer"].update(Panel(f"[dim]{notes}[/dim]", box=box.HORIZONTALS))

        return layout

    def run_once() -> tuple:
        """Execute the full pipeline and return (card, snapshot)."""
        from data.fetcher import Fetcher
        from engine.normalizer import Normalizer
        from engine.axis_scorer import AxisScorer
        from engine.regime_classifier import RegimeClassifier
        from engine.strategy_selector import StrategySelector
        from engine.trade_constructor import TradeConstructor
        from engine.win_probability import WinProbabilityModel

        fetcher = Fetcher()
        snapshot = fetcher.fetch_all()
        normalized = Normalizer().normalize(snapshot)
        axis_scores = AxisScorer().score(normalized)

        vix_val = snapshot.get("VIX").value if snapshot.get("VIX") else None
        vvix_val = snapshot.get("VVIX").value if snapshot.get("VVIX") else None
        vix_term_val = snapshot.get("VIX_TERM").value if snapshot.get("VIX_TERM") else None

        regime = RegimeClassifier().classify(axis_scores, vix_term=vix_term_val, vvix=vvix_val)
        strategy_params = StrategySelector().select(regime, vix=vix_val or 20.0)
        card = TradeConstructor(fetcher).construct(regime, strategy_params)
        card = WinProbabilityModel().compute(card, regime)
        if snapshot.warnings:
            card = card.model_copy(update={"notes": card.notes + snapshot.warnings})
        return card, snapshot

    if once:
        card, snapshot = run_once()
        from rich.console import Console as _C
        _C().print_json(card.model_dump_json(indent=2))
        return

    refresh_seconds = interval * 60
    card, snapshot = None, None

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            start = time.time()
            try:
                card, snapshot = run_once()
            except Exception as e:
                console.print(f"[red]Pipeline error: {e}[/red]")
                time.sleep(60)
                continue

            while True:
                elapsed = time.time() - start
                if elapsed >= refresh_seconds:
                    break
                if card:
                    live.update(build_display(card, snapshot, elapsed))
                time.sleep(1)


def _table_to_str(table: "Table") -> str:
    """Render a Rich Table to a plain string for embedding in a Panel."""
    from io import StringIO
    from rich.console import Console as _C
    buf = StringIO()
    c = _C(file=buf, force_terminal=False, width=60)
    c.print(table)
    return buf.getvalue()


if __name__ == "__main__":
    run_dashboard()
