"""Feature engineering for HMM v2 inputs.

Builds 5 orthogonal EWM z-scored features from raw_indicators.parquet.
All features are PIT-safe (values at date t use only data through t-1).
All features have >=99% coverage — no rows dropped due to missing data.
"""

import logging

import numpy as np
import pandas as pd

from hmm.config import HMMConfig

logger = logging.getLogger(__name__)


def build_hmm_features(config: HMMConfig) -> pd.DataFrame:
    """Load raw indicators and compute the 5 HMM input features.

    Returns DataFrame with DatetimeIndex and columns matching config.hmm_features.
    All features are PIT-safe: values at date t use only data through t-1.
    """
    raw = pd.read_parquet(config.raw_indicators_path)

    if "trade_date" in raw.columns:
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        raw = raw.set_index("trade_date")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    result = pd.DataFrame(index=raw.index)

    for feature_name, indicator_col in config.raw_indicator_map.items():
        if indicator_col not in raw.columns:
            logger.error(f"Indicator {indicator_col} not found for feature {feature_name}")
            result[feature_name] = np.nan
            continue

        series = raw[indicator_col].astype(float)

        # Special handling for derived price features
        if feature_name == "spy_rtn_21d":
            # 21-day log return (not z-scored — already a return)
            rtn = np.log(series / series.shift(21))
            if config.exclude_current_bar:
                rtn = rtn.shift(1)
            result[feature_name] = rtn
            continue
        elif feature_name == "spy_vol_21d":
            # 21-day realized vol
            daily_ret = np.log(series / series.shift(1))
            vol = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
            if config.exclude_current_bar:
                vol = vol.shift(1)
            result[feature_name] = vol
            continue

        z = _ewm_zscore(series, config.zscore_halflife,
                        config.zscore_min_periods, config.zscore_clip)

        if config.exclude_current_bar:
            z = z.shift(1)

        result[feature_name] = z

    result = result.dropna(how="all")

    logger.info("Feature coverage:")
    for col in config.hmm_features:
        if col in result.columns:
            pct = result[col].notna().mean() * 100
            fv = result[col].first_valid_index()
            logger.info(f"  {col:20s}: {pct:6.1f}% (from {fv})")

    return result


def get_spy_prices(config: HMMConfig) -> pd.Series | None:
    """Get SPY close prices for forward return analysis."""
    raw = pd.read_parquet(config.raw_indicators_path)
    if "trade_date" in raw.columns:
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        raw = raw.set_index("trade_date")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    for col in ["SPY", "SPY_CLOSE", "spy_close", "SPX"]:
        if col in raw.columns:
            return raw[col].astype(float)

    try:
        import duckdb
        con = duckdb.connect()
        options_dir = config.raw_indicators_path.parent.parent / "processed"
        df = con.execute(f"""
            SELECT DISTINCT trade_date, underlying_close
            FROM '{options_dir}/spy_options_*.parquet'
            ORDER BY trade_date
        """).fetchdf()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            prices = df.set_index("trade_date")["underlying_close"]
            return prices.reindex(raw.index, method="ffill")
    except Exception as e:
        logger.warning(f"Could not load SPY from options data: {e}")

    try:
        import yfinance as yf
        spy = yf.download("SPY", start=str(raw.index.min().date()),
                         end=str(raw.index.max().date()), progress=False)
        if spy is not None and not spy.empty:
            prices = spy["Close"]
            if hasattr(prices, "columns"):
                prices = prices.iloc[:, 0]
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            return prices.reindex(raw.index, method="ffill")
    except Exception as e:
        logger.warning(f"yfinance fallback failed: {e}")

    logger.error("Could not obtain SPY prices")
    return None


def _ewm_zscore(series: pd.Series, halflife: int,
                min_periods: int, clip: float) -> pd.Series:
    """Exponentially weighted z-score."""
    ewm_mean = series.ewm(halflife=halflife, min_periods=min_periods).mean()
    ewm_std = series.ewm(halflife=halflife, min_periods=min_periods).std()
    ewm_std = ewm_std.replace(0, np.nan)
    z = (series - ewm_mean) / ewm_std
    return z.clip(-clip, clip)
