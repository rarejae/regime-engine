"""Forward return analysis and strategy signal generation."""

import logging

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.similarity import SimilarityResult

logger = logging.getLogger(__name__)


def build_signals(
    results: list[SimilarityResult],
    raw_data: pd.DataFrame,
    config: RegimeConfig,
) -> pd.DataFrame:
    """For each month, compute average forward SPY return from similar months.

    Returns DataFrame with columns: date, signal, similar_avg_fwd, dissimilar_avg_fwd,
    spread, spy_fwd_1m (actual forward return for validation).
    """
    # SPY monthly returns
    if "sp500" in raw_data.columns:
        spy = raw_data["sp500"]
    else:
        logger.error("sp500 not in raw data")
        return pd.DataFrame()

    spy_fwd_1m = spy.pct_change().shift(-1)  # Forward 1-month return

    rows = []
    for r in results:
        # Average forward return from similar historical months
        sim_rets = [spy_fwd_1m.get(d, np.nan) for d in r.similar_dates]
        sim_rets = [x for x in sim_rets if not np.isnan(x)]
        sim_avg = np.mean(sim_rets) if sim_rets else 0.0

        # Average forward return from dissimilar historical months
        dissim_rets = [spy_fwd_1m.get(d, np.nan) for d in r.dissimilar_dates]
        dissim_rets = [x for x in dissim_rets if not np.isnan(x)]
        dissim_avg = np.mean(dissim_rets) if dissim_rets else 0.0

        # Signal: long if similar months had positive returns, short if negative
        signal = 1 if sim_avg > 0 else -1

        actual_fwd = spy_fwd_1m.get(r.target_date, np.nan)

        rows.append({
            "date": r.target_date,
            "signal": signal,
            "similar_avg_fwd": sim_avg,
            "dissimilar_avg_fwd": dissim_avg,
            "spread": sim_avg - dissim_avg,
            "spy_fwd_1m": actual_fwd,
            "n_similar": len(r.similar_dates),
            "n_dissimilar": len(r.dissimilar_dates),
            "min_distance": r.min_distance,
            "closest_date": r.closest_date,
        })

    return pd.DataFrame(rows)


def backtest_signal(signals: pd.DataFrame) -> dict:
    """Backtest the similarity signal on SPY."""
    df = signals.dropna(subset=["spy_fwd_1m"]).copy()

    # Strategy return: signal × actual forward return
    df["strategy_ret"] = df["signal"] * df["spy_fwd_1m"]

    # Buy and hold
    df["bh_ret"] = df["spy_fwd_1m"]

    # Cumulative
    df["strategy_cum"] = (1 + df["strategy_ret"]).cumprod()
    df["bh_cum"] = (1 + df["bh_ret"]).cumprod()

    # Metrics
    n = len(df)
    strat_mean = df["strategy_ret"].mean()
    strat_std = df["strategy_ret"].std()
    bh_mean = df["bh_ret"].mean()
    bh_std = df["bh_ret"].std()

    sharpe_strat = strat_mean / strat_std * np.sqrt(12) if strat_std > 0 else 0
    sharpe_bh = bh_mean / bh_std * np.sqrt(12) if bh_std > 0 else 0

    # Hit rate
    correct = ((df["signal"] > 0) & (df["spy_fwd_1m"] > 0)) | \
              ((df["signal"] < 0) & (df["spy_fwd_1m"] < 0))
    hit_rate = correct.mean()

    # Max drawdown
    cum = df["strategy_cum"]
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Correlation to buy-and-hold
    corr = df["strategy_ret"].corr(df["bh_ret"])

    # Long/short breakdown
    long_months = df[df["signal"] > 0]
    short_months = df[df["signal"] < 0]

    return {
        "n_months": n,
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "strategy_sharpe": sharpe_strat,
        "buyhold_sharpe": sharpe_bh,
        "strategy_ann_ret": strat_mean * 12 * 100,
        "buyhold_ann_ret": bh_mean * 12 * 100,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "corr_to_buyhold": corr,
        "pct_long": (df["signal"] > 0).mean(),
        "long_avg_ret": long_months["strategy_ret"].mean() if len(long_months) else 0,
        "short_avg_ret": short_months["strategy_ret"].mean() if len(short_months) else 0,
        "spread_sharpe": (df["spread"].mean() / df["spread"].std() * np.sqrt(12)
                         if df["spread"].std() > 0 else 0),
        "final_strategy_value": float(df["strategy_cum"].iloc[-1]),
        "final_bh_value": float(df["bh_cum"].iloc[-1]),
    }
