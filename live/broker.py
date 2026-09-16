"""Broker adapters.

Active path: `AlpacaBroker` — the taxable sleeve's automated, all-MOC executor
(REST, no SDK, paper by default). Shelved: `DryRunBroker` / `RobinhoodMCPBroker`
(the abandoned Robinhood-MCP design; kept for reference, not used by V19d live).

Quick check once your paper keys are in .env:
    .venv/bin/python -m live.broker            # account + clock + positions
"""

from __future__ import annotations

import json
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


# ── Alpaca (active: taxable sleeve, all-MOC) ─────────────────────────────────
# Raw REST (no alpaca-py SDK): transparent, auditable, and dodges SDK/py3.14
# compat risk. Paper by default; live requires ALPACA_PAPER=0 AND allow_live=True.
# Fills for V19d always go through the closing auction (type=market, tif=cls), so
# the fill is the official closing print — Alpaca's PFOF routing never touches it.

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    """Any non-2xx from Alpaca, or a refused live-trading guard."""


@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float
    market_value: float
    side: str  # long | short


class AlpacaBroker:
    """Alpaca trading adapter. Reads account/positions and places MOC orders.

    Every V19d order is market-on-close: `place_moc` posts type=market,
    time_in_force=cls, which is eligible ONLY in the primary closing auction and
    must be submitted before 15:50 ET. Whole shares only (CLS rejects fractionals).
    """

    def __init__(self, paper: bool | None = None, key_id: str | None = None,
                 secret: str | None = None, allow_live: bool = False, timeout: int = 20):
        env_paper = os.environ.get("ALPACA_PAPER", "1").strip().lower() not in ("0", "false", "no")
        self.paper = env_paper if paper is None else paper
        self.key = key_id or os.environ.get("ALPACA_API_KEY_ID", "").strip()
        self.secret = secret or os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
        if not self.key or not self.secret:
            raise AlpacaError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env "
                              "(use your PAPER keys first).")
        if not self.paper and not allow_live:
            raise AlpacaError("Refusing LIVE trading: ALPACA_PAPER=0 but allow_live is not set. "
                              "Pass allow_live=True only when you intend real orders.")
        self.base = ALPACA_PAPER_URL if self.paper else ALPACA_LIVE_URL
        self.timeout = timeout
        self._h = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
                   "accept": "application/json"}

    # -- transport --
    def _req(self, method: str, url: str, body: dict | None = None):
        import requests
        r = requests.request(method, url, headers=self._h,
                             data=json.dumps(body) if body is not None else None,
                             timeout=self.timeout)
        if r.status_code == 204 or not r.text:
            if r.ok:
                return None
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text}
        if not r.ok:
            raise AlpacaError(f"{method} {url} -> {r.status_code}: {data}")
        return data

    def _t(self, path: str, body=None, method="GET"):
        return self._req(method, f"{self.base}{path}", body)

    # -- reads --
    def get_account(self) -> dict:
        a = self._t("/v2/account")
        return {"cash": float(a["cash"]), "equity": float(a["equity"]),
                "buying_power": float(a["buying_power"]),
                "status": a.get("status"), "trading_blocked": a.get("trading_blocked"),
                "account_number": a.get("account_number")}

    def get_clock(self) -> dict:
        c = self._t("/v2/clock")
        return {"is_open": bool(c["is_open"]), "timestamp": c["timestamp"],
                "next_open": c["next_open"], "next_close": c["next_close"]}

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in (self._t("/v2/positions") or []):
            out[p["symbol"]] = Position(
                symbol=p["symbol"], shares=float(p["qty"]),
                avg_cost=float(p.get("avg_entry_price", 0) or 0),
                market_value=float(p.get("market_value", 0) or 0),
                side=p.get("side", "long"))
        return out

    def get_quote(self, symbol: str) -> Quote:
        """Latest IEX trade (free feed) — the intraday price for the 3:45 breach check."""
        d = self._req("GET", f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest?feed=iex")
        px = float(d["trade"]["p"]) if d and d.get("trade") else 0.0
        return Quote(symbol=symbol, last=px, bid=px, ask=px)

    # -- orders --
    def place_moc(self, symbol: str, side: str, qty: float,
                  client_order_id: str | None = None) -> OrderResult:
        """Market-on-close: type=market, time_in_force=cls. Whole shares only.
        Submit before 15:50 ET or Alpaca rejects it."""
        side = side.lower()
        if side not in ("buy", "sell"):
            raise AlpacaError(f"side must be buy/sell, got {side!r}")
        whole = int(round(qty))
        if whole < 1:
            raise AlpacaError(f"MOC needs a whole-share qty >= 1, got {qty}")
        if abs(qty - whole) > 1e-6:
            # caller should round upstream; be loud rather than silently change size
            raise AlpacaError(f"MOC qty must be whole shares (CLS rejects fractionals); got {qty}")
        body = {"symbol": symbol, "qty": str(whole), "side": side,
                "type": "market", "time_in_force": "cls"}
        if client_order_id:
            body["client_order_id"] = client_order_id
        o = self._t("/v2/orders", body, method="POST")
        return OrderResult(ok=True, dry_run=False, symbol=symbol, side=side, qty=whole,
                           order_type="moc", limit_px=None, fill_px=None,
                           broker_id=o.get("id"), message=o.get("status", "submitted"))

    def get_order(self, order_id: str) -> dict:
        return self._t(f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self._t(f"/v2/orders/{order_id}", method="DELETE")

    def list_orders(self, status: str = "open") -> list[dict]:
        return self._t(f"/v2/orders?status={status}") or []

    # Protocol compatibility: a plain equity order (used rarely; V19d uses place_moc).
    def place_equity_order(self, symbol: str, side: str, qty: float,
                           order_type: str = "market", limit_px: float | None = None) -> OrderResult:
        if order_type == "moc":
            return self.place_moc(symbol, side, qty)
        body = {"symbol": symbol, "qty": str(qty), "side": side.lower(),
                "type": order_type, "time_in_force": "day"}
        if order_type == "limit" and limit_px:
            body["limit_price"] = str(limit_px)
        o = self._t("/v2/orders", body, method="POST")
        return OrderResult(ok=True, dry_run=False, symbol=symbol, side=side.lower(), qty=qty,
                           order_type=order_type, limit_px=limit_px, fill_px=None,
                           broker_id=o.get("id"), message=o.get("status", "submitted"))


def _self_check() -> None:
    """`python -m live.broker` — verify paper keys, print account/clock/positions."""
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    try:
        b = AlpacaBroker()
    except AlpacaError as e:
        print(f"  ✗ {e}")
        return
    mode = "PAPER" if b.paper else "LIVE"
    print(f"  Alpaca {mode}  ({b.base})")
    acct = b.get_account()
    print(f"  account {acct['account_number']}  status={acct['status']}  "
          f"cash ${acct['cash']:,.2f}  equity ${acct['equity']:,.2f}  "
          f"trading_blocked={acct['trading_blocked']}")
    clk = b.get_clock()
    print(f"  market open={clk['is_open']}  next_close={clk['next_close']}")
    pos = b.get_positions()
    if pos:
        for p in pos.values():
            print(f"    {p.symbol:<5} {p.shares:>10.4f} sh  avg ${p.avg_cost:,.2f}  "
                  f"mkt ${p.market_value:,.2f}")
    else:
        print("    (no open positions)")
    print("  ✓ connectivity OK")


if __name__ == "__main__":
    _self_check()
