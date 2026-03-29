"""Feature engineering for HMM inputs.

Transforms raw macro indicators into the feature set the HMM observes.
All features are PIT-safe (use only data through t-1).
"""

import logging

import numpy as np
import pandas as pd

from hmm.config import HMMConfig

logger = logging.getLogger(__name__)


def build_hmm_features(config: HMMConfig) -> pd.DataFrame:
    """Load raw indicators and SPY prices, compute HMM input features.

    Returns DataFrame with DatetimeIndex and columns matching config.hmm_features.
    All features are PIT-safe: values at date t use only data through t-1.
    """
    raw = pd.read_parquet(config.raw_indicators_path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    result = pd.DataFrame(index=raw.index)

    # SPY prices
    spy_price = get_spy_prices(raw, config)

    if spy_price is not None and len(spy_price.dropna()) > 50:
        # 21-day log return (PIT: shift by 1)
        ret_21d = np.log(spy_price / spy_price.shift(21))
        if config.exclude_current_bar:
            ret_21d = ret_21d.shift(1)
        result["spy_return_21d"] = ret_21d

        # 21-day realized vol (PIT: shift by 1)
        daily_ret = np.log(spy_price / spy_price.shift(1))
        vol_21d = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        if config.exclude_current_bar:
            vol_21d = vol_21d.shift(1)
        result["spy_vol_21d"] = vol_21d

    # Z-scored macro features
    zscore_pairs = {
        "vix_z": "VIX",
        "yield_curve_z": "UST_2S10S",
        "credit_spread_z": "HY_OAS",
        "inflation_z": "CPI_YOY",
    }

    for feature_name, indicator_col in zscore_pairs.items():
        if indicator_col in raw.columns:
            series = raw[indicator_col].astype(float)
            z = _ewm_zscore(series, config.zscore_halflife, config.zscore_clip)
            if config.exclude_current_bar:
                z = z.shift(1)
            result[feature_name] = z
        elif feature_name == "yield_curve_z":
            if "UST_10Y" in raw.columns and "UST_2Y" in raw.columns:
                curve = raw["UST_10Y"].astype(float) - raw["UST_2Y"].astype(float)
                z = _ewm_zscore(curve, config.zscore_halflife, config.zscore_clip)
                if config.exclude_current_bar:
                    z = z.shift(1)
                result[feature_name] = z
            else:
                logger.warning(f"Cannot compute {feature_name}: missing indicator columns")
                result[feature_name] = np.nan
        else:
            logger.warning(f"Indicator {indicator_col} not found for feature {feature_name}")
            result[feature_name] = np.nan

    result = result.dropna(how="all")

    for col in result.columns:
        pct = result[col].notna().mean() * 100
        logger.info(f"  {col}: {pct:.1f}% coverage")

    return result


def get_spy_prices(raw: pd.DataFrame, config: HMMConfig) -> pd.Series | None:
    """Get SPY close prices from options data or raw indicators."""
    # Try loading from options data
    try:
        import duckdb
        con = duckdb.connect()
        df = con.execute(f"""
            SELECT DISTINCT trade_date::DATE as dt,
                   FIRST(underlying_close) as spy
            FROM '{config.options_data_dir}/spy_options_*.parquet'
            WHERE underlying_close > 0
            GROUP BY dt
            ORDER BY dt
        """).fetchdf()
        if not df.empty:
            df["dt"] = pd.to_datetime(df["dt"])
            prices = df.set_index("dt")["spy"]
            return prices.reindex(raw.index, method="ffill")
    except Exception as e:
        logger.warning(f"Could not load SPY prices from options data: {e}")

    # Check raw indicator columns
    for col in ["SPX_PRICE", "SPY", "SPY_CLOSE"]:
        if col in raw.columns:
            return raw[col].astype(float)

    logger.error("Could not obtain SPY prices from any source")
    return None


def _ewm_zscore(series: pd.Series, halflife: int, clip: float) -> pd.Series:
    """Exponentially weighted z-score.

    Recent observations dominate the mean and std computation.
    Halflife=252 means data from 1 year ago has 50% weight.
    """
    min_periods = max(1, halflife // 4)
    ewm_mean = series.ewm(halflife=halflife, min_periods=min_periods).mean()
    ewm_std = series.ewm(halflife=halflife, min_periods=min_periods).std()
    ewm_std = ewm_std.replace(0, np.nan)
    z = (series - ewm_mean) / ewm_std
    return z.clip(-clip, clip)
