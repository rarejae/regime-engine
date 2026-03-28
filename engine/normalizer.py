"""Z-score normalizer for the Macro Regime Engine.

Computes z-scores for each indicator value against its 5-year rolling history.
Clips extreme values to ±z_clip (default 3.0) before passing to axis scoring.

The normalizer maintains a history cache of pandas Series per indicator,
populated on first run by pulling 5 years of data from each source.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from data.models import IndicatorSnapshot, NormalizedIndicator

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
_INDICATORS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "indicators.yaml")

with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

with open(_INDICATORS_PATH) as _f:
    _INDICATORS_CFG = yaml.safe_load(_f)["indicators"]

_NORM_SETTINGS = _SETTINGS["normalization"]
_DEFAULT_Z_CLIP = _NORM_SETTINGS["z_clip"]
_MIN_HISTORY = _NORM_SETTINGS["min_history_periods"]
_LOOKBACK_YEARS = _NORM_SETTINGS["lookback_years"]

# Polarity sign map
_POLARITY_SIGN: dict[str, int] = {
    "risk_on": 1,
    "risk_off": -1,
    "ambiguous": 0,
}


def _polarity_sign(polarity: str) -> int:
    """Return the polarity sign (+1, -1, 0) for a polarity string."""
    return _POLARITY_SIGN.get(polarity, 0)


class HistoryStore:
    """
    Manages per-indicator historical data needed to compute z-scores.

    On first call for an indicator, pulls multi-year history from the source.
    On subsequent calls, only the current value is needed (history already loaded).
    """

    def __init__(self) -> None:
        self._history: dict[str, pd.Series] = {}

    def ensure_history(self, indicator_id: str, current_value: float) -> pd.Series:
        """
        Return the historical series for *indicator_id*, loading it if needed.

        Args:
            indicator_id: Indicator ID from config.
            current_value: The freshly fetched current value to append.

        Returns:
            pd.Series of historical values (at least the current value).
        """
        if indicator_id not in self._history:
            logger.debug(f"Loading history for {indicator_id}")
            series = self._load_history(indicator_id)
            self._history[indicator_id] = series

        # Append current value (deduplicated by using the latest timestamp)
        hist = self._history[indicator_id]
        new_entry = pd.Series([current_value], index=[pd.Timestamp.now()])
        combined = pd.concat([hist, new_entry])
        # Keep only _LOOKBACK_YEARS of data
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * _LOOKBACK_YEARS)
        self._history[indicator_id] = combined[combined.index >= cutoff]
        return self._history[indicator_id]

    def _load_history(self, indicator_id: str) -> pd.Series:
        """
        Pull multi-year raw history for an indicator from the appropriate source.

        Falls back to an empty Series if history cannot be loaded (z-score will
        then use current value as both mean and std=1, returning z=0).

        Args:
            indicator_id: Indicator ID.

        Returns:
            pd.Series with DatetimeIndex.
        """
        cfg = _INDICATORS_CFG.get(indicator_id, {})
        source = cfg.get("source", "")
        years = _LOOKBACK_YEARS

        try:
            if source == "fred":
                from data.sources.fred import FREDSource
                fred = FREDSource()
                series_id = cfg.get("series_id", "")
                derive = cfg.get("derive", "")
                derive_from: list[str] = cfg.get("derive_from", [])

                if derive in ("yoy_pct", "mom_3m_annualized", "delta_12m",
                              "delta_3m_pct", "mom_3ma_thousands", "mom_pct"):
                    # For derived metrics, pull raw series and compute rolling derived values
                    raw = fred.get_history_for_normalization(series_id or derive_from[0], years=years + 1)
                    return self._derive_history(raw, derive, cfg)
                elif derive == "spread_bps" and len(derive_from) == 2:
                    raw_a = fred.get_history_for_normalization(derive_from[0], years=years)
                    raw_b = fred.get_history_for_normalization(derive_from[1], years=years)
                    aligned = pd.DataFrame({"a": raw_a, "b": raw_b}).dropna()
                    return (aligned["a"] - aligned["b"]) * 100
                else:
                    return fred.get_history_for_normalization(series_id, years=years)

            elif source == "yahoo":
                from data.sources.yahoo import YahooSource
                yahoo = YahooSource()
                ticker = cfg.get("ticker", "")
                derive = cfg.get("derive", "")
                derive_from = cfg.get("derive_from", [])

                if derive == "spread" and len(derive_from) == 2:
                    raw_a = yahoo.get_history_for_normalization(derive_from[0], years=years)
                    raw_b = yahoo.get_history_for_normalization(derive_from[1], years=years)
                    aligned = pd.DataFrame({"a": raw_a, "b": raw_b}).dropna()
                    return aligned["a"] - aligned["b"]
                elif derive == "return_3m_pct":
                    prices = yahoo.get_history_for_normalization(ticker, years=years)
                    return prices.pct_change(63).dropna() * 100
                elif derive == "yoy_pct":
                    prices = yahoo.get_history_for_normalization(ticker, years=years + 1)
                    return prices.pct_change(252).dropna() * 100
                elif derive == "up_down_ratio_10d":
                    prices = yahoo.get_history_for_normalization(ticker, years=years)
                    daily_ret = prices.pct_change().dropna()
                    # Rolling 10-day up/down ratio
                    up = daily_ret.rolling(10).apply(lambda x: (x > 0).sum())
                    dn = daily_ret.rolling(10).apply(lambda x: (x <= 0).sum())
                    ratio = (up / dn.replace(0, np.nan)).dropna()
                    return ratio
                elif derive == "trailing_pe":
                    # P/E history not available via yfinance; return empty
                    return pd.Series(dtype=float)
                else:
                    return yahoo.get_history_for_normalization(ticker, years=years)

            elif source == "cboe":
                from data.sources.cboe import CBOESource
                cboe = CBOESource()
                df = cboe.get_historical_put_call()
                if df is not None:
                    s = df.set_index("date")["put_call"]
                    s.index = pd.DatetimeIndex(s.index)
                    return s
                return pd.Series(dtype=float)

            elif source == "gpr":
                from data.sources.gpr import GPRSource
                gpr = GPRSource()
                df = gpr.get_history(years=years)
                s = df.set_index("date")["gpr"]
                s.index = pd.DatetimeIndex(s.index)
                return s

            else:
                return pd.Series(dtype=float)

        except Exception as e:
            logger.warning(f"Could not load history for {indicator_id}: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def _derive_history(raw: pd.Series, derive: str, cfg: dict) -> pd.Series:
        """
        Apply a derivation formula to a raw history series.

        Args:
            raw: Raw pd.Series with DatetimeIndex.
            derive: Derivation formula string.
            cfg: Indicator config dict.

        Returns:
            Derived pd.Series.
        """
        if derive == "yoy_pct":
            return raw.pct_change(12).dropna() * 100  # monthly series
        elif derive == "mom_3m_annualized":
            mom = raw.pct_change(1).dropna()
            return mom.rolling(3).mean().dropna() * 1200  # annualized
        elif derive == "delta_12m":
            return raw.diff(252).dropna() * 100  # bps for daily rate series
        elif derive == "delta_3m_pct":
            return raw.pct_change(63).dropna() * 100
        elif derive == "mom_3ma_thousands":
            diffs = raw.diff(1).dropna()
            return diffs.rolling(3).mean().dropna()
        elif derive == "mom_pct":
            return raw.pct_change(1).dropna() * 100
        else:
            return raw


# ── Public API ─────────────────────────────────────────────────────────────────

class Normalizer:
    """Computes z-scores for each indicator in an IndicatorSnapshot."""

    def __init__(self) -> None:
        self.store = HistoryStore()

    def normalize(self, snapshot: IndicatorSnapshot) -> list[NormalizedIndicator]:
        """
        Normalize all indicators in a snapshot to z-scores.

        Args:
            snapshot: An IndicatorSnapshot from the Fetcher.

        Returns:
            List of NormalizedIndicator objects ready for axis scoring.
        """
        results: list[NormalizedIndicator] = []
        cfg_map = _INDICATORS_CFG

        for ind_id, iv in snapshot.indicators.items():
            cfg = cfg_map.get(ind_id, {})
            z_clip = float(cfg.get("z_clip", _DEFAULT_Z_CLIP))

            try:
                history = self.store.ensure_history(ind_id, iv.value)
                norm = self._compute_z(ind_id, iv.value, history, iv, z_clip)
                results.append(norm)
                logger.debug(
                    f"Normalized {ind_id:20}: raw={iv.value:>10.4f}  "
                    f"z={norm.z_score:>6.3f}  (mean={norm.mean:.4f}, std={norm.std:.4f})"
                )
            except Exception as e:
                logger.warning(f"Normalization failed for {ind_id}: {e}")

        return results

    @staticmethod
    def _compute_z(
        ind_id: str,
        current: float,
        history: pd.Series,
        iv: "IndicatorValue",
        z_clip: float,
    ) -> NormalizedIndicator:
        """
        Compute z-score for one indicator value.

        Args:
            ind_id: Indicator ID.
            current: Current raw value.
            history: Historical series including current value.
            iv: IndicatorValue metadata.
            z_clip: Clipping threshold.

        Returns:
            NormalizedIndicator with z_score, mean, std, polarity_sign.
        """
        clean = history.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean) < _MIN_HISTORY:
            logger.warning(
                f"{ind_id}: only {len(clean)} history points (<{_MIN_HISTORY}); "
                "using z_score=0.0"
            )
            mean = float(current)
            std = 1.0
            z_raw = 0.0
        else:
            mean = float(clean.mean())
            std = float(clean.std())
            if std < 1e-10:
                std = 1.0
                z_raw = 0.0
            else:
                z_raw = (current - mean) / std

        z_clipped = float(np.clip(z_raw, -z_clip, z_clip))

        return NormalizedIndicator(
            id=ind_id,
            raw_value=current,
            z_score=z_clipped,
            z_score_unclipped=z_raw,
            mean=mean,
            std=std,
            polarity_sign=_polarity_sign(iv.polarity),
            weight=iv.weight,
            axis=iv.axis,
            is_proxy=iv.is_proxy,
        )
