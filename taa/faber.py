"""Multi-timeframe Faber SMA trend filter."""

import pandas as pd

SMA_PERIODS = [6, 10, 12]


def compute_trend_scores(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Count how many of 3 SMAs each asset's price is above (0-3).

    Args:
        monthly_prices: Monthly closing prices, indexed by month-start.

    Returns:
        DataFrame of integers (0-3) per asset, indexed by month-start.
        Shifted by 1 month for PIT safety: value at index T is computed
        from prices through month T-1.
    """
    scores = pd.DataFrame(0, index=monthly_prices.index, columns=monthly_prices.columns)
    for period in SMA_PERIODS:
        sma = monthly_prices.rolling(period, min_periods=period).mean()
        scores += (monthly_prices > sma).astype(int)
    return scores.shift(1)  # PIT: use prior month's signal


def apply_faber_filter(trend_scores: dict, baseline: dict) -> tuple[dict, float]:
    """Apply trend filter: strong=full weight, moderate=70%, none=ineligible.

    Args:
        trend_scores: {asset: 0-3} trend strength per asset.
        baseline: {asset: weight} baseline portfolio.

    Returns:
        (eligible_weights, pool_size) — pool is freed capital from filtered assets.
    """
    w = {}
    pool = 0.0
    for a, base_w in baseline.items():
        if a == "cash":
            w[a] = base_w
            continue
        score = trend_scores.get(a, 0)
        if score >= 3:
            w[a] = base_w
        elif score == 2:
            w[a] = base_w * 0.70
            pool += base_w * 0.30
        else:
            w[a] = 0.0
            pool += base_w
    return w, pool
