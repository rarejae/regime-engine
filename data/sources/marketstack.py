"""Marketstack EOD data source.

Requires MARKETSTACK_API_KEY in the environment (.env).
Plan limitation (verified 2026-07-18): history reaches back ~10 years on the
current tier (earliest row ≈ 2016-07). Requests are paginated at 1000 rows.

Results are cached to data/raw/marketstack/ as parquet so backtest reruns
don't burn API quota. Delete the cache file to force a refresh.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.marketstack.com/v1/eod"
CACHE_DIR = Path(__file__).resolve().parent.parent / "raw" / "marketstack"
PAGE_LIMIT = 1000


def _api_key() -> str:
    key = os.environ.get("MARKETSTACK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MARKETSTACK_API_KEY not set — add it to .env")
    return key


def fetch_symbol_eod(symbol: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch full EOD history for one symbol, paginating until exhausted.

    Returns a DataFrame indexed by date with columns [close, adj_close].
    """
    key = _api_key()
    rows = []
    offset = 0
    while True:
        resp = requests.get(BASE_URL, params={
            "access_key": key, "symbols": symbol,
            "date_from": date_from, "date_to": date_to,
            "limit": PAGE_LIMIT, "offset": offset, "sort": "ASC",
        }, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(f"Marketstack error for {symbol}: {payload['error']}")
        data = payload.get("data", [])
        rows.extend(data)
        count = payload["pagination"]["count"]
        offset += count
        # NOTE: pagination 'total' is unreliable (echoes page count) —
        # keep paging until a short page signals exhaustion.
        if count < PAGE_LIMIT:
            break
        time.sleep(0.25)  # stay polite on rate limits

    if not rows:
        return pd.DataFrame(columns=["close", "adj_close", "split_factor"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    # Vendor data quality: Marketstack occasionally returns 0.0 closes
    # (observed Apr/Jun 2026 for SPY/GLD/QLD/SSO). Treat as missing.
    n_bad = int((df["adj_close"] <= 0).sum())
    if n_bad:
        print(f"  NOTE: {symbol} has {n_bad} zero-price rows from Marketstack — treated as missing")
        df.loc[df["adj_close"] <= 0, "adj_close"] = pd.NA
    df["split_factor"] = df["split_factor"].fillna(1.0)
    return df[["close", "adj_close", "split_factor"]]


def _fix_unadjusted_splits(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Back-adjust splits that Marketstack's adj_close failed to apply.

    Vendor inconsistency (verified 2026-07-18): QLD's 2022 2:1 split is
    reflected in adj_close, SSO's is not. For each flagged split day,
    check whether the adjusted series still shows the raw split gap; if
    so, divide all earlier prices by the split factor.
    """
    adj = df["adj_close"].copy()
    for day in df.index[df["split_factor"] != 1.0]:
        f = float(df.loc[day, "split_factor"])
        pos = df.index.get_loc(day)
        if pos == 0 or f <= 0:
            continue
        prev = adj.iloc[pos - 1]
        # If prev/current ≈ split factor, the split was NOT adjusted out.
        if abs(prev / (adj.loc[day] * f) - 1) < 0.10:
            adj.loc[:df.index[pos - 1]] = adj.loc[:df.index[pos - 1]] / f
            print(f"  NOTE: applied missing {f:g}:1 split adjustment for "
                  f"{symbol} on {day.date()}")
    return adj


def load_adj_closes(symbols: list[str], date_from: str = "2015-01-01",
                    date_to: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Adjusted closes for symbols as a single DataFrame, cached to parquet."""
    date_to = date_to or pd.Timestamp.now().strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"eod_{'_'.join(sorted(symbols))}.parquet"

    if cache_file.exists() and not refresh:
        cached = pd.read_parquet(cache_file)
        # Cache is fresh if it extends to within 3 days of the requested end
        if (pd.Timestamp(date_to) - cached.index.max()).days <= 3:
            return cached

    frames = {}
    for sym in symbols:
        df = fetch_symbol_eod(sym, date_from, date_to)
        if df.empty:
            print(f"  WARNING: Marketstack returned no data for {sym}")
            continue
        frames[sym] = _fix_unadjusted_splits(df, sym)
        print(f"  Marketstack {sym}: {len(df)} rows, "
              f"{df.index.min().date()} → {df.index.max().date()}")

    out = pd.DataFrame(frames).sort_index()
    out.to_parquet(cache_file)
    return out
