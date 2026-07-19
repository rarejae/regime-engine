"""Paper twin ledger — intended vs fill, slip in bps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime"
LEDGER = RUNTIME / "ledger.jsonl"
SIGNALS = RUNTIME / "signals.jsonl"
PAPER_NAV = RUNTIME / "paper_nav.csv"


def _append_jsonl(path: Path, row: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def log_signal(eval_result: dict, actions: list[dict]) -> None:
    _append_jsonl(SIGNALS, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "eval": eval_result,
        "actions": actions,
    })


def log_fill(
    *,
    sleeve: str,
    action: str,
    ticker: str | None,
    qty: float,
    intended_px: float,
    fill_px: float,
    dry_run: bool,
    note: str = "",
) -> dict:
    slip_bps = None
    if intended_px and fill_px and intended_px > 0:
        # sell: positive slip if fill < intended; buy: positive if fill > intended
        if action in ("CB_SELL", "SELL"):
            slip_bps = (intended_px - fill_px) / intended_px * 1e4
        else:
            slip_bps = (fill_px - intended_px) / intended_px * 1e4
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "sleeve": sleeve,
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "intended_px": intended_px,
        "fill_px": fill_px,
        "slip_bps": slip_bps,
        "dry_run": dry_run,
        "note": note,
    }
    _append_jsonl(LEDGER, row)
    return row


def read_ledger(n: int = 50) -> list[dict]:
    if not LEDGER.exists():
        return []
    lines = LEDGER.read_text().strip().splitlines()
    return [json.loads(x) for x in lines[-n:]]
