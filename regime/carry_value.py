"""Carry and value signals per asset."""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ASSETS = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]


def _fred(sid, start="1960-01-01"):
    try:
        from fredapi import Fred
        key = os.environ.get("FRED_API_KEY")
        if key:
            return Fred(api_key=key).get_series(sid, observation_start=start).dropna()
    except Exception:
        pass
    return None


def compute_carry(prices_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute monthly carry per asset. Shifted by 1 for PIT safety.

    Returns DataFrame indexed by month-start with one column per asset.
    """
    frames = {}

    # T-bill rate (monthly)
    tb3 = _fred("DTB3", "1998-01-01")
    if tb3 is None:
        tb3 = _fred("TB3MS", "1998-01-01")
    if tb3 is not None:
        tb3.index = pd.to_datetime(tb3.index)
        tb3_m = tb3.resample("MS").last() / 1200  # monthly rate
    else:
        tb3_m = pd.Series(0.003, index=pd.date_range("1998-01", "2026-12", freq="MS"))

    # VGLT carry: GS20 - DTB3 (term premium)
    gs20 = _fred("GS20", "1998-01-01")
    if gs20 is None:
        gs20 = _fred("GS10", "1998-01-01")  # fallback to 10Y
    if gs20 is not None:
        gs20.index = pd.to_datetime(gs20.index)
        gs20_m = gs20.resample("MS").last() / 1200
        frames["VGLT"] = (gs20_m - tb3_m).dropna()

    # IVV carry: dividend yield - risk-free
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        divs = spy.dividends
        if divs is not None and len(divs) > 0:
            divs.index = pd.to_datetime(divs.index).tz_localize(None)
            div_12m = divs.rolling("365D").sum().resample("MS").last()
            spy_price = yf.download("SPY", start="1998-01-01", progress=False)["Close"]
            if hasattr(spy_price, "columns"):
                spy_price = spy_price.iloc[:, 0]
            spy_price.index = pd.to_datetime(spy_price.index).tz_localize(None)
            spy_m = spy_price.resample("MS").last()
            div_yield_m = (div_12m / spy_m / 12).dropna()
            frames["IVV"] = (div_yield_m - tb3_m).dropna()
    except Exception as e:
        logger.warning(f"IVV carry failed: {e}")

    # QQQ carry: same approach
    if "IVV" in frames:
        # QQQ dividend yield is ~0.5% vs IVV ~1.8%, approximate
        frames["QQQ"] = frames["IVV"] * 0.3  # rough ratio

    # IAU carry: -risk-free (gold has zero yield)
    frames["IAU"] = -tb3_m

    # DBC carry: set to 0 (commodity roll yield is unreliable from available data)
    frames["DBC"] = pd.Series(0.0, index=tb3_m.index)

    # Cash carry: risk-free rate itself
    frames["cash"] = tb3_m

    df = pd.DataFrame(frames).sort_index()
    # Shift by 1 month for PIT safety
    return df.shift(1)


def compute_value(prices_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute monthly value z-scores per asset. Shifted by 1 for PIT safety.

    Returns DataFrame indexed by month-start with one column per asset.
    """
    frames = {}

    # IVV value: -z-score of CAPE ratio
    # Use trailing PE from yfinance as CAPE proxy
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        info = spy.info or {}
        # Can't get historical PE easily; use earnings yield proxy
        # Use FRED earnings series
        pe = _fred("MULTPL/SP500_PE_RATIO_MONTH", "1970-01-01")  # may not exist
        if pe is None:
            # Approximate: S&P price / 10Y trailing earnings from FRED
            sp_price = _fred("SPASTT01USM661N", "1970-01-01")
            sp_earnings = _fred("A053RC1Q027SBEA", "1970-01-01")  # quarterly earnings
            if sp_price is not None and sp_earnings is not None:
                sp_m = sp_price.resample("MS").last()
                earn_m = sp_earnings.resample("MS").last().ffill()
                # 10-year trailing earnings sum (rough)
                earn_10y = earn_m.rolling(40, min_periods=20).sum()  # 40 quarters
                cape_proxy = sp_m / earn_10y
                cape_proxy = cape_proxy.dropna()
                if len(cape_proxy) > 120:
                    mu = cape_proxy.rolling(120, min_periods=60).mean()
                    sd = cape_proxy.rolling(120, min_periods=60).std().replace(0, np.nan)
                    frames["IVV"] = -((cape_proxy - mu) / sd).clip(-3, 3)
    except Exception as e:
        logger.warning(f"IVV value failed: {e}")

    # QQQ value: use IVV as proxy if no separate data
    if "IVV" in frames:
        frames["QQQ"] = frames["IVV"]  # correlated

    # VGLT value: real yield (higher = cheaper = positive value)
    gs20 = _fred("GS20", "1970-01-01")
    if gs20 is None:
        gs20 = _fred("GS10", "1970-01-01")
    cpi = _fred("CPIAUCSL", "1970-01-01")
    if gs20 is not None and cpi is not None:
        gs20.index = pd.to_datetime(gs20.index)
        cpi.index = pd.to_datetime(cpi.index)
        gs20_m = gs20.resample("MS").last()
        cpi_m = cpi.resample("MS").last()
        cpi_yoy = cpi_m.pct_change(12) * 100
        real_yield = (gs20_m - cpi_yoy).dropna()
        if len(real_yield) > 120:
            mu = real_yield.rolling(120, min_periods=60).mean()
            sd = real_yield.rolling(120, min_periods=60).std().replace(0, np.nan)
            frames["VGLT"] = ((real_yield - mu) / sd).clip(-3, 3)  # positive = cheap

    # IAU value: -z-score of real gold price (high = expensive)
    gold = _fred("WPU1017", "1970-01-01")  # PPI precious metals
    if gold is not None and cpi is not None:
        gold.index = pd.to_datetime(gold.index)
        gold_m = gold.resample("MS").last()
        cpi_m2 = cpi.resample("MS").last().reindex(gold_m.index, method="ffill")
        real_gold = (gold_m / cpi_m2 * 100).dropna()
        if len(real_gold) > 240:
            mu = real_gold.rolling(240, min_periods=120).mean()  # 20yr
            sd = real_gold.rolling(240, min_periods=120).std().replace(0, np.nan)
            frames["IAU"] = -((real_gold - mu) / sd).clip(-3, 3)

    # DBC value: -z-score of real commodity price
    ppi = _fred("PPIACO", "1970-01-01")
    if ppi is not None and cpi is not None:
        ppi.index = pd.to_datetime(ppi.index)
        ppi_m = ppi.resample("MS").last()
        cpi_m3 = cpi.resample("MS").last().reindex(ppi_m.index, method="ffill")
        real_comm = (ppi_m / cpi_m3 * 100).dropna()
        if len(real_comm) > 240:
            mu = real_comm.rolling(240, min_periods=120).mean()
            sd = real_comm.rolling(240, min_periods=120).std().replace(0, np.nan)
            frames["DBC"] = -((real_comm - mu) / sd).clip(-3, 3)

    # Cash value: always 0 (cash has no value signal)
    if frames:
        idx = list(frames.values())[0].index
        frames["cash"] = pd.Series(0.0, index=idx)

    df = pd.DataFrame(frames).sort_index()
    return df.shift(1)  # PIT safety
