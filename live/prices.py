"""Swappable market-data source for the V19d signal.

The signal (daily adjusted closes → 126/200/252 SMAs → the trade decision) defaults
to **Tiingo**: official consolidated closes, deep history, clean corporate actions —
the right basis for a close-based signal whose live orders fill in the official
closing auction. Alpaca supplies the intraday 3:45 cutoff quote and execution
(live/execute.py) and is deliberately NOT the signal source: its free feed is
IEX-only (single venue), so its "close" is not the official closing print.

Switch sources with PRICE_SOURCE=tiingo|yfinance|cache (default tiingo).
Returns a DataFrame indexed by trading day with a column per ticker (adjusted close).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIGNAL_TICKERS = ["QQQ", "IVV", "IAU", "QLD", "SSO"]
DEFAULT_START = "2015-01-01"  # ample history for 252-day SMAs (QLD/SSO exist from 2006)


def load_signal_prices(source: str | None = None, start: str = DEFAULT_START,
                       tickers: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    src = (source or os.environ.get("PRICE_SOURCE", "tiingo")).lower()
    tickers = tickers or SIGNAL_TICKERS
    if src == "tiingo":
        df = _from_tiingo(tickers, start, refresh)
    elif src == "yfinance":
        df = _from_yfinance(tickers, start)
    elif src == "cache":
        df = _from_cache(tickers)
    else:
        raise SystemExit(f"unknown PRICE_SOURCE '{src}' — use tiingo|yfinance|cache")
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all").sort_index()


def _from_tiingo(tickers: list[str], start: str, refresh: bool) -> pd.DataFrame:
    from data.sources.tiingo import load_adj_closes
    df = load_adj_closes(tickers, date_from=start, refresh=refresh)
    missing = [t for t in ("QQQ", "IVV", "IAU") if t not in df.columns]
    if missing:
        raise SystemExit(f"Tiingo missing signal tickers {missing} — cannot compute SMAs")
    return df


def _from_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    out = {}
    for t in tickers:
        d = yf.download(t, start=start, progress=False, auto_adjust=True)
        if d is None or getattr(d, "empty", True):
            continue
        p = d["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        out[t] = p.dropna()
    return pd.DataFrame(out)


def _from_cache(tickers: list[str]) -> pd.DataFrame:
    """Legacy fallback: the yfinance parquet under the marketstack-verification experiment."""
    cache = ROOT / "experiments" / "v19d_marketstack_verification" / "yf_cache.parquet"
    if not cache.exists():
        raise SystemExit(f"cache source selected but {cache} is missing")
    raw = pd.read_parquet(cache)
    colmap = {"QQQ": "QQQ", "SPY": "IVV", "GLD": "IAU", "QLD": "QLD", "SSO": "SSO"}
    return pd.DataFrame({dst: raw[src] for src, dst in colmap.items() if src in raw.columns})
