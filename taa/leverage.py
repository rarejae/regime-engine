"""Synthetic leveraged ETF return computation and validation."""

import os
import numpy as np
import pandas as pd


def compute_synthetic_leveraged(
    underlying_daily_ret: pd.Series,
    rfr_daily: pd.Series,
    leverage: float = 2.0,
    annual_expense: float = 0.0091,
) -> pd.Series:
    """Compute synthetic leveraged ETF daily returns.

    synthetic_ret = leverage × underlying_ret - (leverage - 1) × rfr_daily - expense_daily
    """
    expense_daily = annual_expense / 252
    # Align rfr to underlying
    rfr = rfr_daily.reindex(underlying_daily_ret.index, method="ffill").fillna(0)
    synth = leverage * underlying_daily_ret - (leverage - 1) * rfr - expense_daily
    return synth


def build_synthetic_etfs(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic SSO (2x IVV) and QLD (2x QQQ).

    Args:
        daily_ret: DataFrame with IVV, QQQ, and cash columns.

    Returns:
        DataFrame with SSO and QLD daily return columns added.
    """
    # Daily risk-free rate
    key = os.environ.get("FRED_API_KEY")
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    if key:
        try:
            from fredapi import Fred
            tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
            if tb is not None:
                tb.index = pd.to_datetime(tb.index)
                rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)
        except Exception:
            pass

    result = daily_ret.copy()

    if "IVV" in daily_ret.columns:
        result["SSO"] = compute_synthetic_leveraged(
            daily_ret["IVV"], rfr_daily, leverage=2.0, annual_expense=0.0091
        )

    if "QQQ" in daily_ret.columns:
        result["QLD"] = compute_synthetic_leveraged(
            daily_ret["QQQ"], rfr_daily, leverage=2.0, annual_expense=0.0089
        )

    return result


def validate_synthetic(synthetic_ret: pd.Series, actual_ticker: str,
                        name: str, start: str = "2006-06-01") -> dict:
    """Compare synthetic leveraged ETF to actual ETF from yfinance."""
    import yfinance as yf

    d = yf.download(actual_ticker, start=start, progress=False)
    if d is None or d.empty:
        return {"status": "FAIL", "reason": f"Could not download {actual_ticker}"}

    actual_price = d["Close"]
    if hasattr(actual_price, "columns"):
        actual_price = actual_price.iloc[:, 0]
    actual_price.index = pd.to_datetime(actual_price.index).tz_localize(None)
    actual_ret = actual_price.pct_change().dropna()

    # Align
    common = synthetic_ret.index.intersection(actual_ret.index)
    if len(common) < 252:
        return {"status": "FAIL", "reason": f"Only {len(common)} overlapping days"}

    s = synthetic_ret.reindex(common).dropna()
    a = actual_ret.reindex(common).dropna()
    common2 = s.index.intersection(a.index)
    s = s.reindex(common2)
    a = a.reindex(common2)

    # Metrics
    s_ann = s.mean() * 252
    a_ann = a.mean() * 252
    corr = s.corr(a)

    s_cum = (1 + s).cumprod()
    a_cum = (1 + a).cumprod()
    s_dd = ((s_cum - s_cum.expanding().max()) / s_cum.expanding().max()).min()
    a_dd = ((a_cum - a_cum.expanding().max()) / a_cum.expanding().max()).min()

    # Annual returns comparison
    annual = {}
    for yr in [2007, 2008, 2009, 2020, 2022]:
        sy = s[s.index.year == yr]
        ay = a[a.index.year == yr]
        if len(sy) > 20 and len(ay) > 20:
            annual[yr] = {
                "synthetic": float((1 + sy).prod() - 1),
                "actual": float((1 + ay).prod() - 1),
            }

    ret_diff = abs(s_ann - a_ann)
    dd_diff = abs(s_dd - a_dd)

    return {
        "status": "PASS" if ret_diff < 0.01 and corr > 0.99 else "CHECK",
        "name": name,
        "n_days": len(common2),
        "synth_ann_ret": float(s_ann),
        "actual_ann_ret": float(a_ann),
        "ret_diff": float(ret_diff),
        "correlation": float(corr),
        "synth_max_dd": float(s_dd),
        "actual_max_dd": float(a_dd),
        "dd_diff": float(dd_diff),
        "annual": annual,
    }
