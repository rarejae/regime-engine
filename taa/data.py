"""Data loading for TAA module — independent of regime/ module."""

import os
import pandas as pd
import numpy as np


def load_monthly_macro() -> pd.DataFrame:
    """Load monthly macro history for Harvey similarity engine."""
    return pd.read_parquet("data/macro/monthly_history.parquet")


def load_monthly_asset_returns() -> pd.DataFrame:
    """Load monthly ETF returns (IVV, QQQ, VGLT, IAU, DBC, cash)."""
    df = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    keep = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]
    return df[[c for c in keep if c in df.columns]]


def load_daily_etf_returns() -> pd.DataFrame:
    """Fetch daily ETF returns."""
    import yfinance as yf

    frames = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our] = p.pct_change()

    key = os.environ.get("FRED_API_KEY")
    if key:
        from fredapi import Fred
        try:
            tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
            if tb is not None:
                tb.index = pd.to_datetime(tb.index)
                frames["cash"] = tb / 36000
        except Exception:
            pass

    return pd.DataFrame(frames).sort_index()


def load_monthly_prices() -> pd.DataFrame:
    """Load monthly closing prices for SMA computation."""
    import yfinance as yf

    frames = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our] = p
    df = pd.DataFrame(frames).sort_index()
    return df.resample("MS").last()


def load_realized_vols(daily_prices: pd.DataFrame = None) -> pd.DataFrame:
    """Compute 63-day trailing realized vol, resampled monthly, PIT-shifted."""
    if daily_prices is None:
        import yfinance as yf
        frames = {}
        for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
            d = yf.download(ticker, start="1998-01-01", progress=False)
            if d is not None and not d.empty:
                p = d["Close"]
                if hasattr(p, "columns"): p = p.iloc[:, 0]
                p.index = pd.to_datetime(p.index).tz_localize(None)
                frames[our] = p
        daily_prices = pd.DataFrame(frames).sort_index()

    rets = daily_prices.pct_change()
    rvol = rets.rolling(63, min_periods=30).std() * np.sqrt(252)
    return rvol.resample("MS").last().shift(1)  # PIT
