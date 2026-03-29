"""Harvey-Mulliner variable transformation: 12-month change, rolling z-score, winsorize."""

import numpy as np
import pandas as pd

from regime.config import RegimeConfig


def transform_variables(raw: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Apply Harvey methodology: 12-month change → rolling 10yr std z-score → winsorize.

    For each variable:
      change_t = val_t - val_{t-12}
      std_t = rolling_std(change, window=120 months)
      z_t = change_t / std_t
      z_t = clip(z_t, -3, +3)
    """
    result = pd.DataFrame(index=raw.index)

    for col in raw.columns:
        series = raw[col].astype(float)

        # 12-month change
        change = series - series.shift(config.change_lookback)

        # Rolling 10-year std of 12-month changes
        rolling_std = change.rolling(
            window=config.zscore_window,
            min_periods=config.zscore_window // 2,
        ).std()

        # Z-score
        z = change / rolling_std.replace(0, np.nan)

        # Winsorize
        z = z.clip(-config.winsorize_clip, config.winsorize_clip)

        result[f"{col}_z"] = z

    # Also keep raw change for inspection
    for col in raw.columns:
        result[f"{col}_chg"] = raw[col] - raw[col].shift(config.change_lookback)

    return result


def get_valid_zscored(transformed: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Return only z-scored columns, drop rows with any NaN."""
    z_cols = [c for c in transformed.columns if c.endswith("_z")]
    return transformed[z_cols].dropna()
