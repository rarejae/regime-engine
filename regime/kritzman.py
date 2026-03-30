"""Kritzman turbulence index + absorption ratio computation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_daily_basket() -> pd.DataFrame:
    """Fetch daily cross-asset returns for turbulence/absorption computation."""
    import yfinance as yf
    import os
    from fredapi import Fred

    frames = {}

    # SPY
    d = yf.download("SPY", start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        frames["spy"] = np.log(p / p.shift(1))

    # TLT (from 2002) — pre-2002 use yield-derived bond returns
    tlt = yf.download("TLT", start="2002-07-01", progress=False)
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        fred = Fred(api_key=key)
        gs10 = fred.get_series("DGS10", observation_start="1998-01-01")
        gs10.index = pd.to_datetime(gs10.index)
        bond_ret = -7.0 * gs10.diff() / 100 + gs10.shift(1) / 36500
        bond_ret = bond_ret.dropna()
        if tlt is not None and not tlt.empty:
            tp = tlt["Close"]
            if hasattr(tp, "columns"): tp = tp.iloc[:, 0]
            tp.index = pd.to_datetime(tp.index).tz_localize(None)
            tlt_ret = np.log(tp / tp.shift(1))
            cutoff = tlt_ret.index.min()
            bond_ret = pd.concat([bond_ret[bond_ret.index < cutoff], tlt_ret]).sort_index()
        frames["bonds"] = bond_ret

    # VIX daily change
    vix = yf.download("^VIX", start="1998-01-01", progress=False)
    if vix is not None and not vix.empty:
        vp = vix["Close"]
        if hasattr(vp, "columns"): vp = vp.iloc[:, 0]
        vp.index = pd.to_datetime(vp.index).tz_localize(None)
        frames["vix_chg"] = vp.diff()

    # GLD (from 2004) — pre-2004 use gold futures
    gld = yf.download("GLD", start="2004-11-01", progress=False)
    gc = yf.download("GC=F", start="2000-01-01", progress=False)
    gold_parts = []
    if gc is not None and not gc.empty:
        gcp = gc["Close"]
        if hasattr(gcp, "columns"): gcp = gcp.iloc[:, 0]
        gcp.index = pd.to_datetime(gcp.index).tz_localize(None)
        gold_parts.append(np.log(gcp / gcp.shift(1)))
    if gld is not None and not gld.empty:
        gp = gld["Close"]
        if hasattr(gp, "columns"): gp = gp.iloc[:, 0]
        gp.index = pd.to_datetime(gp.index).tz_localize(None)
        gold_parts.append(np.log(gp / gp.shift(1)))
    if gold_parts:
        gold = gold_parts[0].copy()
        if len(gold_parts) > 1:
            cutoff = gold_parts[1].index.min()
            gold = pd.concat([gold[gold.index < cutoff], gold_parts[1]]).sort_index()
        frames["gold"] = gold

    # HY OAS daily change from FRED
    if key:
        hy = fred.get_series("BAMLH0A0HYM2", observation_start="1998-01-01")
        if hy is not None:
            hy.index = pd.to_datetime(hy.index)
            frames["hy_chg"] = hy.diff()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info(f"Daily basket: {df.shape[0]} days, {df.shape[1]} assets, "
                f"from {df.index.min().date()}")
    return df


def compute_turbulence(basket: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Mahalanobis distance turbulence index."""
    clean = basket.dropna()
    turb = pd.Series(np.nan, index=clean.index, dtype=float)

    for i in range(lookback, len(clean)):
        window = clean.iloc[i - lookback:i]
        mu = window.mean().values
        cov = window.cov().values
        try:
            obs = clean.iloc[i].values
            inv_cov = np.linalg.inv(cov)
            diff = obs - mu
            d = float(np.sqrt(diff @ inv_cov @ diff))
            turb.iloc[i] = d
        except (np.linalg.LinAlgError, ValueError):
            continue

    turb = turb.dropna()
    # Smooth: sqrt + 21-day EMA
    smoothed = np.sqrt(turb).ewm(span=21).mean()
    return smoothed


def compute_turbulence_pctl(turb_smooth: pd.Series, window: int = 252) -> pd.Series:
    """Trailing percentile rank of smoothed turbulence."""
    return turb_smooth.rolling(window, min_periods=63).rank(pct=True)


def compute_absorption_ratio(basket: pd.DataFrame, lookback: int = 500,
                              n_components: int | None = None) -> pd.Series:
    """Absorption ratio: variance explained by top eigenvectors / total variance."""
    clean = basket.dropna()
    n_assets = clean.shape[1]
    if n_components is None:
        n_components = max(1, n_assets // 5)

    ar = pd.Series(np.nan, index=clean.index, dtype=float)

    for i in range(lookback, len(clean)):
        window = clean.iloc[i - lookback:i]
        cov = window.cov().values
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # descending
            total_var = eigenvalues.sum()
            top_var = eigenvalues[:n_components].sum()
            ar.iloc[i] = top_var / total_var if total_var > 0 else np.nan
        except np.linalg.LinAlgError:
            continue

    return ar.dropna()


def compute_ar_zscore(ar: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score of absorption ratio."""
    mu = ar.rolling(window, min_periods=63).mean()
    sd = ar.rolling(window, min_periods=63).std().replace(0, np.nan)
    return (ar - mu) / sd


def apply_turbulence_scaling(
    target_weights: dict,
    turb_pctl: float,
    risk_assets: list[str] = None,
    defensive_assets: list[str] = None,
) -> dict:
    """Scale risk assets down based on turbulence percentile.

    <80th: scale=1.0, 80-95th: linear 1.0→0.5, >95th: scale=0.3
    Freed capital goes to cash.
    """
    if risk_assets is None:
        risk_assets = ["IVV", "QQQ", "DBC", "VNQ"]
    if defensive_assets is None:
        defensive_assets = ["VGLT", "IAU"]

    if turb_pctl < 0.80:
        scale = 1.0
    elif turb_pctl < 0.95:
        scale = 1.0 - 0.5 * (turb_pctl - 0.80) / 0.15
    else:
        scale = 0.3

    w = dict(target_weights)
    freed = 0.0
    for a in risk_assets:
        if a in w:
            original = w[a]
            w[a] = original * scale
            freed += original - w[a]

    w["cash"] = w.get("cash", 0) + freed
    return w


def apply_absorption_constraints(
    weights: dict,
    ar_z: float,
    ar_falling: bool,
    turb_falling: bool,
    recovery_progress: float = 0.0,
) -> dict:
    """Apply absorption ratio constraints on equity and defensive floors.

    AR z < 0.5: no constraints
    AR z 0.5-1.5: max equity 30%, min defensives 25%, min cash 15%
    AR z > 1.5: max equity 10%, min defensives 30%, min cash 25%

    When AR and turbulence are both falling: release constraints progressively.
    """
    equity = ["IVV", "QQQ"]
    defensives = ["VGLT", "IAU"]

    if ar_z < 0.5:
        return weights

    # Determine constraint level
    if ar_z > 1.5:
        max_eq = 0.10; min_def = 0.30; min_cash = 0.25
    else:
        max_eq = 0.30; min_def = 0.25; min_cash = 0.15

    # Release constraints during recovery
    if ar_falling and turb_falling and recovery_progress > 0:
        blend = min(recovery_progress, 1.0)
        max_eq = max_eq + (1.0 - max_eq) * blend
        min_def = min_def * (1.0 - blend)
        min_cash = min_cash * (1.0 - blend)

    w = dict(weights)

    # Cap equity
    eq_total = sum(w.get(a, 0) for a in equity)
    if eq_total > max_eq and eq_total > 0:
        ratio = max_eq / eq_total
        freed = 0.0
        for a in equity:
            old = w.get(a, 0)
            w[a] = old * ratio
            freed += old - w[a]
        w["cash"] = w.get("cash", 0) + freed

    # Floor defensives
    def_total = sum(w.get(a, 0) for a in defensives)
    if def_total < min_def:
        deficit = min_def - def_total
        # Take from non-defensive, non-cash proportionally
        others = [a for a in w if a not in defensives and a != "cash" and w[a] > 0]
        total_others = sum(w[a] for a in others)
        if total_others > deficit:
            for a in others:
                take = deficit * (w[a] / total_others)
                w[a] -= take
            # Distribute to defensives equally
            for a in defensives:
                w[a] = w.get(a, 0) + deficit / len(defensives)

    # Floor cash
    if w.get("cash", 0) < min_cash:
        deficit = min_cash - w.get("cash", 0)
        others = [a for a in w if a != "cash" and w[a] > 0]
        total_others = sum(w[a] for a in others)
        if total_others > deficit:
            for a in others:
                take = deficit * (w[a] / total_others)
                w[a] -= take
            w["cash"] = min_cash

    return w
