"""
Macro Regime Engine — CLI Entry Point

Full pipeline:
  1. Fetch all macro indicators
  2. Normalize to z-scores
  3. Score composite axes (GROWTH, INFLATION, LIQUIDITY, RISK)
  4. Classify regime + modifiers + conviction
  5. Select strategy
  6. Construct specific SPY vertical spread
  7. Compute win probability
  8. Print TradeCard

Usage:
    python main.py                     # full run with all indicators
    python main.py --debug             # verbose DEBUG logging
    python main.py --indicators VIX HY_OAS CLAIMS_4WK  # subset fetch
    python main.py --no-cache          # bypass cache (force fresh fetch)
    python main.py --json              # output TradeCard as JSON
"""

from __future__ import annotations

import json
import os
import sys

import click
from dotenv import load_dotenv
from loguru import logger

# Load .env before any imports that read env vars
load_dotenv()


@click.command()
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
@click.option(
    "--indicators",
    "-i",
    multiple=True,
    default=None,
    help="Fetch only specific indicator IDs (repeatable)",
)
@click.option("--no-cache", is_flag=True, help="Bypass cache and fetch all data fresh")
@click.option("--json", "output_json", is_flag=True, help="Output TradeCard as JSON")
def main(debug: bool, indicators: tuple[str, ...], no_cache: bool, output_json: bool) -> None:
    """Macro Regime Engine: classify macro regime and output optimal SPY spread."""

    from utils.logging import setup_logging
    setup_logging("DEBUG" if debug else None)

    if no_cache:
        from data.cache import clear_all
        clear_all()
        logger.warning("Cache cleared — fetching all data fresh")

    # ── Phase 1: Fetch ────────────────────────────────────────────────────────
    from data.fetcher import Fetcher
    fetcher = Fetcher()

    ind_list = list(indicators) if indicators else None
    logger.info("=" * 60)
    logger.info("MACRO REGIME ENGINE — Starting fetch")
    logger.info("=" * 60)

    snapshot = fetcher.fetch_all(ind_list)

    if not snapshot.indicators:
        logger.error("No indicators fetched. Check API keys and network connectivity.")
        sys.exit(1)

    # ── Phase 2: Normalize ────────────────────────────────────────────────────
    from engine.normalizer import Normalizer
    normalizer = Normalizer()
    normalized = normalizer.normalize(snapshot)
    logger.info(f"Normalized {len(normalized)} indicators")

    # ── Phase 3: Axis scoring ─────────────────────────────────────────────────
    from engine.axis_scorer import AxisScorer
    scorer = AxisScorer()
    axis_scores = scorer.score(normalized)

    # ── Phase 4: Regime classification ───────────────────────────────────────
    from engine.regime_classifier import RegimeClassifier

    # Pull VIX and VVIX for modifier computation (if available)
    vix_indicator = snapshot.get("VIX")
    vvix_indicator = snapshot.get("VVIX")
    vix_term_indicator = snapshot.get("VIX_TERM")

    vix_val = vix_indicator.value if vix_indicator else None
    vvix_val = vvix_indicator.value if vvix_indicator else None
    vix_term_val = vix_term_indicator.value if vix_term_indicator else None

    classifier = RegimeClassifier()
    regime = classifier.classify(axis_scores, vix_term=vix_term_val, vvix=vvix_val)

    # ── Phase 5: Strategy selection ───────────────────────────────────────────
    from engine.strategy_selector import StrategySelector
    selector = StrategySelector()
    vix_for_strategy = vix_val or 20.0
    strategy_params = selector.select(regime, vix=vix_for_strategy)

    # ── Phase 6: Trade construction ───────────────────────────────────────────
    from engine.trade_constructor import TradeConstructor
    constructor = TradeConstructor(fetcher)

    try:
        card = constructor.construct(regime, strategy_params)
    except Exception as e:
        logger.error(f"Trade construction failed: {e}")
        logger.info("Printing regime analysis without trade card")
        _print_regime_summary(regime, axis_scores, snapshot)
        sys.exit(1)

    # ── Phase 7: Win probability ──────────────────────────────────────────────
    from engine.win_probability import WinProbabilityModel
    win_model = WinProbabilityModel()
    card = win_model.compute(card, regime)

    # Attach snapshot warnings to card notes
    if snapshot.warnings:
        card = card.model_copy(update={"notes": card.notes + snapshot.warnings})

    # ── Phase 8: Output ───────────────────────────────────────────────────────
    if output_json:
        click.echo(card.model_dump_json(indent=2))
    else:
        _print_trade_card(card)

    logger.info("Pipeline complete")


def _print_trade_card(card: "TradeCard") -> None:
    """Pretty-print the TradeCard to stdout using rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        from rich.text import Text

        console = Console()

        # Regime panel
        regime_color = {
            "GOLDILOCKS": "green",
            "REFLATION": "yellow",
            "STAGFLATION": "red",
            "DEFLATION": "bright_red",
            "MIXED": "cyan",
        }.get(card.regime, "white")

        regime_text = Text()
        regime_text.append(f"  {card.regime}", style=f"bold {regime_color}")
        if card.modifiers:
            regime_text.append(f"  [{', '.join(card.modifiers)}]", style="dim")

        # Axis scores table
        axis_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        axis_table.add_column("Axis", style="cyan")
        axis_table.add_column("Score", justify="right")
        axis_table.add_column("Signal", justify="center")

        for axis, score in card.axis_scores.items():
            bar = "▇" * int(abs(score) / 10) if abs(score) >= 5 else "·"
            signal = f"[green]+{score:.1f}[/green]" if score >= 0 else f"[red]{score:.1f}[/red]"
            axis_table.add_row(axis, signal, bar)

        # Trade table
        trade_table = Table(box=box.SIMPLE, show_header=False)
        trade_table.add_column("Field", style="bold")
        trade_table.add_column("Value")

        is_credit = card.strategy_type == "credit"
        credit_label = "Net Credit" if is_credit else "Net Debit"
        credit_val = f"${abs(card.estimated_credit):.2f}/share  (${abs(card.estimated_credit)*100:.0f}/contract)"

        trade_table.add_row("Strategy", f"[bold]{card.strategy}[/bold]")
        trade_table.add_row("Underlying", card.underlying)
        trade_table.add_row("Short Strike / Δ", f"{card.short_strike:.0f}  (Δ {card.short_delta:.3f})")
        trade_table.add_row("Long Strike / Δ", f"{card.long_strike:.0f}  (Δ {card.long_delta:.3f})")
        trade_table.add_row("Width", f"${card.width:.0f}")
        trade_table.add_row("Expiry", f"{card.expiry}  ({card.dte}d)")
        trade_table.add_row(credit_label, credit_val)
        trade_table.add_row("Max Risk", f"${card.max_risk:.2f}/share  (${card.max_risk*100:.0f}/contract)")
        trade_table.add_row("Credit / Width", f"{card.credit_to_width:.1%}")
        trade_table.add_row("Breakeven", f"${card.breakeven:.2f}")
        trade_table.add_row(
            "Win Probability",
            f"[bold green]{card.win_probability*100:.1f}%[/bold green]" if card.win_probability > 0.60
            else f"[bold yellow]{card.win_probability*100:.1f}%[/bold yellow]",
        )
        trade_table.add_row("Conviction", f"{card.conviction:.0f}/100  ({card.transition})")

        # Management table
        mgmt_table = Table(box=box.SIMPLE, show_header=False)
        mgmt_table.add_column("Rule", style="bold")
        mgmt_table.add_column("Value")
        for k, v in card.management.items():
            mgmt_table.add_row(k.replace("_", " ").title(), str(v))

        console.print()
        console.print(Panel(regime_text, title="[bold]MACRO REGIME[/bold]", border_style=regime_color))
        console.print(Panel(axis_table, title="[bold]AXIS SCORES[/bold]", border_style="cyan"))
        console.print(Panel(trade_table, title="[bold]TRADE CARD[/bold]", border_style="green"))
        console.print(Panel(mgmt_table, title="[bold]TRADE MANAGEMENT[/bold]", border_style="yellow"))

        if card.notes:
            notes_text = "\n".join(f"  ⚠  {n}" for n in card.notes)
            console.print(Panel(notes_text, title="[bold]NOTES & PROXIES[/bold]", border_style="dim"))

        console.print(f"\n[dim]Generated: {card.timestamp}[/dim]\n")

    except ImportError:
        # Fallback: plain text
        print(_trade_card_plain(card))


def _print_regime_summary(regime, axis_scores, snapshot) -> None:
    """Print a minimal regime summary when trade construction fails."""
    print(f"\nRegime:     {regime.primary}")
    print(f"Modifiers:  {regime.modifiers}")
    print(f"Conviction: {regime.conviction:.0f}/100")
    print(f"Transition: {regime.transition}")
    print(f"\nAxis Scores:")
    for ax in ("GROWTH", "INFLATION", "LIQUIDITY", "RISK"):
        score = getattr(axis_scores, ax)
        print(f"  {ax:12} {score:+.1f}")


def _trade_card_plain(card: "TradeCard") -> str:
    """Return a plain-text TradeCard representation."""
    lines = [
        "",
        f"{'='*50}",
        f"  REGIME:    {card.regime}  {card.modifiers}",
        f"  STRATEGY:  {card.strategy}  (conviction: {card.conviction:.0f})",
        f"{'='*50}",
        f"  SPY {card.short_strike:.0f}/{card.long_strike:.0f}  exp {card.expiry} ({card.dte}d)",
        f"  Credit:    ${abs(card.estimated_credit):.2f}/share",
        f"  Max Risk:  ${card.max_risk:.2f}/share",
        f"  Breakeven: ${card.breakeven:.2f}",
        f"  Win Prob:  {card.win_probability*100:.1f}%",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
