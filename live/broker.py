"""Broker adapters. Default dry-run; Robinhood MCP when connected."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Quote:
    symbol: str
    last: float
    bid: float | None = None
    ask: float | None = None


@dataclass
class OrderResult:
    ok: bool
    dry_run: bool
    symbol: str
    side: str
    qty: float
    order_type: str
    limit_px: float | None
    fill_px: float | None
    broker_id: str | None
    message: str


class Broker(Protocol):
    def get_quote(self, symbol: str) -> Quote: ...
    def place_equity_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
        limit_px: float | None = None,
    ) -> OrderResult: ...


class DryRunBroker:
    """Simulates fills at last price (or limit). Never sends orders."""

    def __init__(self, quotes: dict[str, float] | None = None):
        self.quotes = quotes or {}

    def set_quote(self, symbol: str, last: float) -> None:
        self.quotes[symbol] = last

    def get_quote(self, symbol: str) -> Quote:
        px = float(self.quotes.get(symbol, 0.0))
        return Quote(symbol=symbol, last=px, bid=px, ask=px)

    def place_equity_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
        limit_px: float | None = None,
    ) -> OrderResult:
        q = self.get_quote(symbol)
        fill = limit_px if (order_type == "limit" and limit_px) else q.last
        return OrderResult(
            ok=True,
            dry_run=True,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_px=limit_px,
            fill_px=fill,
            broker_id=f"DRY-{symbol}-{side}",
            message="dry-run fill at quote/limit",
        )


class RobinhoodMCPBroker:
    """Placeholder until Robinhood Trading MCP is connected in Cursor.

    Endpoint: https://agent.robinhood.com/mcp/trading
    When LIVE_TRADING=1 and MCP tools are available, wire CallMcpTool here.
    """

    def __init__(self, fallback: DryRunBroker | None = None):
        self.fallback = fallback or DryRunBroker()
        self.live = os.environ.get("LIVE_TRADING", "").strip() in ("1", "true", "yes")

    def get_quote(self, symbol: str) -> Quote:
        # MCP not connected in this workspace yet — use fallback quotes
        return self.fallback.get_quote(symbol)

    def place_equity_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
        limit_px: float | None = None,
    ) -> OrderResult:
        if not self.live:
            return self.fallback.place_equity_order(symbol, side, qty, order_type, limit_px)
        return OrderResult(
            ok=False,
            dry_run=False,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_px=limit_px,
            fill_px=None,
            broker_id=None,
            message=(
                "LIVE_TRADING set but Robinhood MCP is not connected in this workspace. "
                "Add https://agent.robinhood.com/mcp/trading via OAuth, then rewire this adapter."
            ),
        )


def get_broker(quotes: dict[str, float] | None = None) -> Broker:
    dry = os.environ.get("DRY_RUN", "true").lower() not in ("0", "false", "no")
    fb = DryRunBroker(quotes)
    if dry:
        return fb
    return RobinhoodMCPBroker(fallback=fb)
