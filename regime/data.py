"""Fetch and cache monthly macro history from FRED and yfinance."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import RegimeConfig

logger = logging.getLogger(__name__)


def fetch_monthly_history(config: RegimeConfig, force: bool = False) -> pd.DataFrame:
    """Fetch all 7 raw variables at monthly frequency. Cache to parquet."""
    if config.monthly_history_path.exists() and not force:
        existing = pd.read_parquet(config.monthly_history_path)
        logger.info(f"Using cached monthly history: {existing.shape}")
        return existing

    config.monthly_history_path.parent.mkdir(parents=True, exist_ok=True)
    start = config.fred_start

    frames = {}

    # 1. S&P 500 (monthly) — OECD composite from FRED goes back to 1960
    sp = _fred("SPASTT01USM661N", start)
    if sp is None or len(sp) < 200:
        sp = _yfinance_monthly("^GSPC", "1955-01-01")
    if sp is None or len(sp) < 200:
        sp = _fred("SP500", start)
    if sp is not None:
        frames["sp500"] = sp.resample("MS").last().dropna()
        logger.info(f"  sp500: {len(frames['sp500'])} months from {frames['sp500'].index.min().date()}")

    # 2. Yield curve = GS10 - TB3MS
    gs10 = _fred("GS10", start)
    tb3ms = _fred("TB3MS", start)
    if gs10 is not None and tb3ms is not None:
        gs10_m = gs10.resample("MS").last()
        tb3ms_m = tb3ms.resample("MS").last()
        yc = (gs10_m - tb3ms_m).dropna()
        frames["yield_curve"] = yc
        logger.info(f"  yield_curve: {len(yc)} months")

    # 3. Oil — WTISPLC (Spot WTI) goes back to 1955
    oil = _fred("WTISPLC", start)
    if oil is None or len(oil) < 200:
        oil = _fred("MCOILWTICO", start)  # fallback
    if oil is not None:
        frames["oil"] = oil.resample("MS").last().dropna()
        logger.info(f"  oil: {len(frames['oil'])} months from {frames['oil'].index.min().date()}")

    # 4. Copper — WPU102301 (PPI Copper & Brass) goes back to 1957
    cu = _fred("WPU102301", start)
    if cu is None or len(cu) < 200:
        cu = _fred("PCOPPUSDM", start)  # global price fallback (from 1980)
    if cu is None or len(cu) < 100:
        cu = _yfinance_monthly("HG=F", "1960-01-01")
    if cu is not None:
        frames["copper"] = cu.resample("MS").last().dropna()
        logger.info(f"  copper: {len(frames['copper'])} months from {frames['copper'].index.min().date()}")

    # 5. T-bill yield
    if tb3ms is not None:
        frames["tbill"] = tb3ms.resample("MS").last().dropna()
        logger.info(f"  tbill: {len(frames['tbill'])} months")

    # 6. Volatility (VIX from 1990, realized vol before)
    vix = _yfinance_monthly("^VIX", "1990-01-01")
    sp_daily = _yfinance_daily("^GSPC", start)
    if sp_daily is not None:
        # Realized vol: 21-day rolling std of daily returns × sqrt(252)
        daily_ret = np.log(sp_daily / sp_daily.shift(1))
        rv = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
        rv_monthly = rv.resample("MS").mean().dropna()

        if vix is not None and len(vix) > 0:
            vix_m = vix.resample("MS").mean().dropna()
            # Merge: VIX from 1990, realized vol before
            vol = rv_monthly.copy()
            vol.loc[vix_m.index] = vix_m
            frames["vol"] = vol
        else:
            frames["vol"] = rv_monthly
        logger.info(f"  vol: {len(frames['vol'])} months from {frames['vol'].index.min().date()}")

    # 7. Stock-bond correlation (3Y rolling)
    if sp_daily is not None:
        gs10_daily = _fred("DGS10", start)
        if gs10_daily is not None:
            # Bond return proxy: -duration × yield change. Approx: -7 * daily change
            bond_ret = -7 * gs10_daily.diff()
            stock_ret = np.log(sp_daily / sp_daily.shift(1))
            # Align
            both = pd.DataFrame({"stock": stock_ret, "bond": bond_ret}).dropna()
            corr_3y = both["stock"].rolling(756, min_periods=504).corr(both["bond"])
            corr_monthly = corr_3y.resample("MS").last().dropna()
            frames["stock_bond_corr"] = corr_monthly
            logger.info(f"  stock_bond_corr: {len(corr_monthly)} months from {corr_monthly.index.min().date()}")

    # Combine
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_parquet(config.monthly_history_path)
    logger.info(f"Saved: {df.shape[0]} rows × {df.shape[1]} cols to {config.monthly_history_path}")
    return df


def _fred(series_id: str, start: str) -> pd.Series | None:
    try:
        from fredapi import Fred
        key = os.environ.get("FRED_API_KEY")
        if key:
            fred = Fred(api_key=key)
            s = fred.get_series(series_id, observation_start=start)
            s.index = pd.to_datetime(s.index)
            return s.dropna()
    except Exception as e:
        logger.debug(f"fredapi failed for {series_id}: {e}")

    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return s
    except Exception as e:
        logger.warning(f"FRED CSV failed for {series_id}: {e}")
        return None


def _yfinance_monthly(ticker: str, start: str) -> pd.Series | None:
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, interval="1mo", progress=False)
        if data is None or data.empty:
            return None
        prices = data["Close"]
        if hasattr(prices, "columns"):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        return prices
    except Exception as e:
        logger.warning(f"yfinance monthly failed for {ticker}: {e}")
        return None


def _yfinance_daily(ticker: str, start: str) -> pd.Series | None:
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, progress=False)
        if data is None or data.empty:
            return None
        prices = data["Close"]
        if hasattr(prices, "columns"):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        return prices
    except Exception as e:
        logger.warning(f"yfinance daily failed for {ticker}: {e}")
        return None
