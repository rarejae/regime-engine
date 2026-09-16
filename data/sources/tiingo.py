"""Tiingo EOD data source — the V19d signal's market data.

Tiingo's free personal tier gives official (consolidated) split- and dividend-
adjusted daily closes with deep history (QQQ→1999) and correct corporate actions
(the IAU 2021-05-24 2:1 split is handled). That is the right basis for a close-
based signal whose live fills settle in the official closing auction — unlike a
single-venue (IEX) feed. Requires TIINGO_API_KEY in .env.

adjClose is already fully split+dividend adjusted, so unlike the Marketstack
loader there is no manual split back-adjustment to do. Results cache to
data/raw/tiingo/ as parquet; a cache within 3 days of the requested end is reused
(delete the file or pass refresh=True to force a pull). Free tier ≈ 50 req/hour —
a daily 5-ticker pull is trivially under that.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.tiingo.com/tiingo/daily"
CACHE_DIR = Path(__file__).resolve().parent.parent / "raw" / "tiingo"


def _api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TIINGO_API_KEY not set — add it to .env")
    return key


def _get(url: str, params: dict, tries: int = 5):
    """GET with a 429-aware wait (free-tier hourly cap)."""
    params = {**params, "token": _api_key()}
    for i in range(tries):
        r = requests.get(url, params=params, headers={"Content-Type": "application/json"}, timeout=30)
        if r.status_code == 429:
            wait = 60 * (i + 1)
            print(f"  Tiingo 429 rate-limited; waiting {wait}s (attempt {i+1}/{tries})…")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Tiingo rate-limited past retries")


def fetch_symbol_eod(symbol: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Full adjusted EOD history for one symbol → DataFrame[date] with [close, adj_close]."""
    rows = _get(f"{BASE_URL}/{symbol}/prices",
                {"startDate": date_from, "endDate": date_to, "columns": "close,adjClose"})
    if not rows:
        return pd.DataFrame(columns=["close", "adj_close"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"adjClose": "adj_close"})
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    n_bad = int((df["adj_close"] <= 0).sum())
    if n_bad:
        print(f"  NOTE: {symbol} has {n_bad} zero/negative-price rows from Tiingo — treated as missing")
        df.loc[df["adj_close"] <= 0, "adj_close"] = pd.NA
    return df[["close", "adj_close"]]


def load_adj_closes(symbols: list[str], date_from: str = "2015-01-01",
                    date_to: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Adjusted closes for symbols as one DataFrame (columns=symbols), cached to parquet."""
    date_to = date_to or pd.Timestamp.now().strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"eod_{'_'.join(sorted(symbols))}.parquet"

    if cache_file.exists() and not refresh:
        cached = pd.read_parquet(cache_file)
        if len(cached) and (pd.Timestamp(date_to) - cached.index.max()).days <= 3:
            return cached

    frames = {}
    for sym in symbols:
        df = fetch_symbol_eod(sym, date_from, date_to)
        if df.empty:
            print(f"  WARNING: Tiingo returned no data for {sym}")
            continue
        frames[sym] = df["adj_close"]
        print(f"  Tiingo {sym}: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")

    out = pd.DataFrame(frames).sort_index()
    if len(out):
        out.to_parquet(cache_file)
    return out
