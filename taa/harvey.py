"""Harvey-Mulliner similarity engine — clean implementation."""

import numpy as np
import pandas as pd


def compute_zscore_variables(raw_monthly: pd.DataFrame,
                              change_lookback: int = 12,
                              zscore_window: int = 120,
                              clip: float = 3.0) -> pd.DataFrame:
    """Transform raw monthly indicators to z-scored 12-month changes.

    Returns DataFrame of z-scores, shifted by 1 month for PIT safety.
    """
    result = pd.DataFrame(index=raw_monthly.index)
    for col in raw_monthly.columns:
        change = raw_monthly[col] - raw_monthly[col].shift(change_lookback)
        rolling_std = change.rolling(zscore_window, min_periods=zscore_window // 2).std()
        z = (change / rolling_std.replace(0, np.nan)).clip(-clip, clip)
        result[f"{col}_z"] = z
    return result.shift(1).dropna(how="all")  # PIT: shift by 1 month


def find_similar_months(z_data: pd.DataFrame,
                         target_date: pd.Timestamp,
                         exclude_months: int = 36,
                         similar_pctl: float = 0.15) -> tuple[list, float]:
    """Find historically similar months by Euclidean distance.

    Args:
        z_data: Z-scored variables (already shifted for PIT).
        target_date: Assessment date.
        exclude_months: Trailing months to exclude (anti-momentum).
        similar_pctl: Bottom percentile = "similar" months.

    Returns:
        (similar_dates, min_distance)
    """
    if target_date not in z_data.index:
        raise ValueError(f"{target_date} not in z_data")

    target_z = z_data.loc[target_date].values
    cutoff = target_date - pd.DateOffset(months=exclude_months)
    candidates = z_data[z_data.index <= cutoff]

    if len(candidates) == 0:
        raise ValueError(f"No candidates before {cutoff}")

    diffs = np.array(candidates.values - target_z, dtype=np.float64)
    distances = pd.Series(np.sqrt(np.sum(diffs ** 2, axis=1)), index=candidates.index)

    threshold = distances.quantile(similar_pctl)
    similar = distances[distances <= threshold].sort_values()

    return similar.index.tolist(), float(distances.min())


def compute_expected_returns(similar_dates: list,
                              asset_returns_fwd: pd.DataFrame,
                              assets: list) -> dict:
    """Compute mean forward 1-month return from similar months per asset.

    Args:
        similar_dates: List of historically similar month dates.
        asset_returns_fwd: Asset returns shifted by -1 (forward return lookup).
        assets: List of asset names.

    Returns:
        {asset: mean_forward_return}
    """
    er = {}
    for a in assets:
        if a not in asset_returns_fwd.columns:
            er[a] = 0.0
            continue
        rets = [asset_returns_fwd.loc[d, a] for d in similar_dates
                if d in asset_returns_fwd.index and pd.notna(asset_returns_fwd.loc[d, a])]
        er[a] = float(np.mean(rets)) if rets else 0.0
    return er
