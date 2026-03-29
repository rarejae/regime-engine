"""Multi-asset return fetching and allocation strategies."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ASSET_RETURNS_PATH = Path("data/macro/asset_returns.parquet")

ASSETS = ["us_equity", "us_bonds", "gold", "commodities", "intl_equity", "tips", "cash"]


def fetch_asset_returns(force: bool = False) -> pd.DataFrame:
    """Fetch monthly total return series for 7 asset classes. Cache to parquet."""
    if ASSET_RETURNS_PATH.exists() and not force:
        logger.info("Using cached asset returns")
        return pd.read_parquet(ASSET_RETURNS_PATH)

    ASSET_RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = {}

    # 1. US equity — SPY total return approx from price
    spy = _yf_monthly_return("SPY", "1993-01-01")
    sp_fred = _fred_series("SPASTT01USM661N", "1957-01-01")
    if sp_fred is not None:
        sp_ret = sp_fred.resample("MS").last().pct_change().dropna()
        if spy is not None and len(spy) > 0:
            # Splice: FRED before ETF start, SPY after
            cutoff = spy.index.min()
            combined = pd.concat([sp_ret[sp_ret.index < cutoff], spy]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames["us_equity"] = combined
        else:
            frames["us_equity"] = sp_ret
    elif spy is not None:
        frames["us_equity"] = spy
    _log_asset("us_equity", frames)

    # 2. US long-term treasuries — bond total return from yield changes
    gs10 = _fred_series("GS10", "1960-01-01")
    if gs10 is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        dy = gs10_m.diff()
        coupon = gs10_m.shift(1) / 1200  # monthly yield
        bond_ret = -7.0 * dy / 100 + coupon  # duration × yield change + coupon
        bond_ret = bond_ret.dropna()
        tlt = _yf_monthly_return("TLT", "2002-07-01")
        if tlt is not None and len(tlt) > 0:
            cutoff = tlt.index.min()
            combined = pd.concat([bond_ret[bond_ret.index < cutoff], tlt]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames["us_bonds"] = combined
        else:
            frames["us_bonds"] = bond_ret
    _log_asset("us_bonds", frames)

    # 3. Gold — FRED gold price, then GLD
    gold_fred = _fred_series("GOLDAMGBD228NLBM", "1968-01-01")
    if gold_fred is not None:
        gold_m = gold_fred.resample("MS").last().pct_change().dropna()
        gld = _yf_monthly_return("GLD", "2004-11-01")
        if gld is not None:
            cutoff = gld.index.min()
            combined = pd.concat([gold_m[gold_m.index < cutoff], gld]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames["gold"] = combined
        else:
            frames["gold"] = gold_m
    _log_asset("gold", frames)

    # 4. Commodities — PPI all commodities as proxy, then DBC
    ppi = _fred_series("PPIACO", "1960-01-01")
    if ppi is not None:
        ppi_m = ppi.resample("MS").last().pct_change().dropna()
        dbc = _yf_monthly_return("DBC", "2006-02-01")
        if dbc is not None:
            cutoff = dbc.index.min()
            combined = pd.concat([ppi_m[ppi_m.index < cutoff], dbc]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames["commodities"] = combined
        else:
            frames["commodities"] = ppi_m
    _log_asset("commodities", frames)

    # 5. Intl developed equity — EFA, with pre-ETF FRED proxy
    efa = _yf_monthly_return("EFA", "2001-08-01")
    # Pre-ETF: use OECD total share prices for advanced economies
    intl_fred = _fred_series("SPASTT01OEM661N", "1960-01-01")  # OECD Europe
    if intl_fred is not None:
        intl_ret = intl_fred.resample("MS").last().pct_change().dropna()
        if efa is not None:
            cutoff = efa.index.min()
            combined = pd.concat([intl_ret[intl_ret.index < cutoff], efa]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames["intl_equity"] = combined
        else:
            frames["intl_equity"] = intl_ret
    elif efa is not None:
        frames["intl_equity"] = efa
    _log_asset("intl_equity", frames)

    # 6. TIPS/inflation-linked — real yield proxy: tbill minus CPI
    tb3ms = _fred_series("TB3MS", "1960-01-01")
    cpi = _fred_series("CPIAUCSL", "1960-01-01")
    if tb3ms is not None and cpi is not None:
        tb_m = tb3ms.resample("MS").last()
        cpi_m = cpi.resample("MS").last()
        cpi_yoy = cpi_m.pct_change(12) * 100
        real_yield = (tb_m - cpi_yoy) / 1200  # monthly real return proxy
        real_yield = real_yield.dropna()
        tip = _yf_monthly_return("TIP", "2003-12-01")
        if tip is not None:
            cutoff = tip.index.min()
            combined = pd.concat([real_yield[real_yield.index < cutoff], tip]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames["tips"] = combined
        else:
            frames["tips"] = real_yield
    _log_asset("tips", frames)

    # 7. Cash — T-bill rate / 12
    if tb3ms is not None:
        cash_ret = tb3ms.resample("MS").last() / 1200
        frames["cash"] = cash_ret.dropna()
    _log_asset("cash", frames)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_parquet(ASSET_RETURNS_PATH)
    logger.info(f"Saved: {df.shape[0]} months × {df.shape[1]} assets")
    return df


def _log_asset(name, frames):
    if name in frames:
        s = frames[name]
        logger.info(f"  {name}: {s.notna().sum()} months from {s.first_valid_index()}")


def _fred_series(sid, start):
    try:
        from fredapi import Fred
        key = os.environ.get("FRED_API_KEY")
        if key:
            fred = Fred(api_key=key)
            s = fred.get_series(sid, observation_start=start)
            s.index = pd.to_datetime(s.index)
            return s.dropna()
    except Exception:
        pass
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        return None


def _yf_monthly_return(ticker, start):
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, interval="1mo", progress=False)
        if data is None or data.empty:
            return None
        prices = data["Close"]
        if hasattr(prices, "columns"):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        ret = prices.pct_change().dropna()
        ret.index = ret.index.to_period("M").to_timestamp()
        return ret
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker}: {e}")
        return None


# ── Allocation strategies ─────────────────────────────────────────────────────

def allocate_similarity_weighted(expected_returns: dict, max_single: float = 0.40) -> dict:
    """Strategy 1: Long-only, weight proportional to expected return."""
    pos = {a: max(0, r) for a, r in expected_returns.items()}
    total = sum(pos.values())
    if total == 0:
        # All negative — go 100% cash
        return {a: (1.0 if a == "cash" else 0.0) for a in expected_returns}
    weights = {a: v / total for a, v in pos.items()}
    # Cap at max_single
    for a in weights:
        weights[a] = min(weights[a], max_single)
    # Renormalize
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def allocate_risk_parity_tilt(expected_returns: dict, vol: dict,
                               tilt_strength: float = 0.5,
                               max_single: float = 0.40) -> dict:
    """Strategy 2: Risk parity baseline with regime tilt."""
    # Inverse vol weights
    inv_vol = {a: 1.0 / max(v, 0.001) for a, v in vol.items() if a in expected_returns}
    total_iv = sum(inv_vol.values())
    base = {a: v / total_iv for a, v in inv_vol.items()}

    # Tilt
    tilted = {}
    for a in base:
        er = expected_returns.get(a, 0)
        tilt = 1.0 + tilt_strength * np.sign(er) * min(abs(er) * 20, 1.0)
        tilted[a] = base[a] * max(tilt, 0.1)

    # Cap and normalize
    for a in tilted:
        tilted[a] = min(tilted[a], max_single)
    total = sum(tilted.values())
    return {a: w / total for a, w in tilted.items()}


def allocate_top_n(expected_returns: dict, n: int = 3) -> dict:
    """Strategy 3: Equal-weight top N assets."""
    ranked = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)
    top = [a for a, _ in ranked[:n]]
    w = 1.0 / n
    return {a: (w if a in top else 0.0) for a in expected_returns}


def allocate_long_short(similar_er: dict, dissimilar_er: dict) -> dict:
    """Strategy 4: Long top 3 similar, short bottom 3 dissimilar."""
    # Long top 3 by similar expected return
    sim_ranked = sorted(similar_er.items(), key=lambda x: x[1], reverse=True)
    long_assets = [a for a, _ in sim_ranked[:3]]

    # Short bottom 3 by dissimilar expected return (but never short cash)
    dissim_ranked = sorted(dissimilar_er.items(), key=lambda x: x[1])
    short_assets = [a for a, _ in dissim_ranked if a != "cash"][:3]

    weights = {a: 0.0 for a in similar_er}
    for a in long_assets:
        weights[a] += 1.0 / 3
    for a in short_assets:
        weights[a] -= 1.0 / 3
    return weights
