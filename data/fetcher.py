"""Indicator Fetcher — orchestrates all data sources into a unified IndicatorSnapshot.

This module is the central entry point for the data layer. It:
  1. Reads indicator config from config/indicators.yaml
  2. Dispatches each indicator to the appropriate source module
  3. Applies derivation formulas where needed
  4. Respects per-cadence caching via data/cache.py
  5. Handles failures gracefully: logs warnings, marks indicators as missing,
     and proportionally re-weights the remaining indicators on each axis.

Usage:
    from data.fetcher import Fetcher
    snapshot = Fetcher().fetch_all()
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import yaml
from loguru import logger

from data.cache import cached
from data.models import IndicatorSnapshot, IndicatorValue

# ── Lazy source singletons ────────────────────────────────────────────────────
_fred: Any = None
_yahoo: Any = None
_bls: Any = None
_cboe: Any = None
_atlanta: Any = None
_aaii: Any = None
_gpr: Any = None


def _get_fred():
    global _fred
    if _fred is None:
        from data.sources.fred import FREDSource
        _fred = FREDSource()
    return _fred


def _get_yahoo():
    global _yahoo
    if _yahoo is None:
        from data.sources.yahoo import YahooSource
        _yahoo = YahooSource()
    return _yahoo


def _get_bls():
    global _bls
    if _bls is None:
        from data.sources.bls import BLSSource
        _bls = BLSSource()
    return _bls


def _get_cboe():
    global _cboe
    if _cboe is None:
        from data.sources.cboe import CBOESource
        _cboe = CBOESource()
    return _cboe


def _get_atlanta():
    global _atlanta
    if _atlanta is None:
        from data.sources.atlanta_fed import AtlantaFedSource
        _atlanta = AtlantaFedSource()
    return _atlanta


def _get_aaii():
    global _aaii
    if _aaii is None:
        from data.sources.aaii import AAIISource
        _aaii = AAIISource()
    return _aaii


def _get_gpr():
    global _gpr
    if _gpr is None:
        from data.sources.gpr import GPRSource
        _gpr = GPRSource()
    return _gpr


# ── Config loader ──────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "indicators.yaml")
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["indicators"]


def _load_settings() -> dict:
    with open(_SETTINGS_PATH) as f:
        return yaml.safe_load(f)


# ── Per-indicator fetch dispatch ───────────────────────────────────────────────

def _fetch_fred_indicator(ind_id: str, cfg: dict) -> tuple[float, str, bool]:
    """
    Fetch a single FRED-sourced indicator value.

    Returns:
        Tuple of (value, proxy_note, is_proxy).
    """
    fred = _get_fred()
    series_id = cfg.get("series_id", "")
    derive = cfg.get("derive", "")
    derive_from: list[str] = cfg.get("derive_from", [])

    if derive == "yoy_pct":
        return fred.get_yoy_pct(series_id), "", False
    elif derive == "mom_3m_annualized":
        return fred.get_mom_3m_annualized(series_id), "", False
    elif derive == "delta_12m":
        return fred.get_delta_12m(series_id), "", False
    elif derive == "delta_3m_pct":
        return fred.get_delta_3m_pct(series_id), "", False
    elif derive == "spread_bps" and len(derive_from) == 2:
        return fred.get_spread_bps(derive_from[0], derive_from[1]), "", False
    elif derive == "mom_3ma_thousands":
        return fred.get_mom_3ma_thousands(series_id), "", False
    elif derive == "mom_pct":
        return fred.get_mom_pct(series_id), "", False
    else:
        # Plain latest value
        return fred.get_latest(series_id), "", False


def _fetch_yahoo_indicator(ind_id: str, cfg: dict) -> tuple[float, str, bool]:
    """
    Fetch a single Yahoo Finance indicator value.

    Returns:
        Tuple of (value, proxy_note, is_proxy).
    """
    yahoo = _get_yahoo()
    ticker = cfg.get("ticker", "")
    derive = cfg.get("derive", "")
    derive_from: list[str] = cfg.get("derive_from", [])

    if derive == "spread" and len(derive_from) == 2:
        # VIX term structure: VIX3M - VIX
        a = yahoo.get_current_price(derive_from[0])
        b = yahoo.get_current_price(derive_from[1])
        return a - b, "", False
    elif derive == "return_3m_pct":
        return yahoo.get_return_3m_pct(ticker), "", False
    elif derive == "yoy_pct":
        return yahoo.get_yoy_pct(ticker), "", False
    elif derive == "up_down_ratio_10d":
        return yahoo.get_up_down_ratio_10d(ticker), "", False
    elif derive == "trailing_pe":
        pe = yahoo.get_trailing_pe(ticker)
        if pe is None:
            raise ValueError(f"P/E unavailable for {ticker}")
        return pe, "Trailing P/E used as forward P/E proxy", True
    else:
        return yahoo.get_current_price(ticker), "", False


def _fetch_indicator(ind_id: str, cfg: dict) -> Optional[IndicatorValue]:
    """
    Dispatch fetch for one indicator, apply caching, return IndicatorValue or None.

    Args:
        ind_id: Indicator ID string, e.g. "VIX".
        cfg: Indicator config dict from indicators.yaml.

    Returns:
        IndicatorValue on success, None on failure (after logging).
    """
    source = cfg.get("source", "")
    cadence = cfg.get("refresh", "daily")
    is_optional = cfg.get("optional", False)
    cache_key = f"indicator::{ind_id}"

    def _do_fetch() -> tuple[float, str, bool]:
        if source == "fred":
            return _fetch_fred_indicator(ind_id, cfg)
        elif source == "yahoo":
            return _fetch_yahoo_indicator(ind_id, cfg)
        elif source == "bls":
            # BLS is backup — try FRED first
            bls = _get_bls()
            if ind_id == "CPI_YOY":
                return bls.get_cpi_yoy(), "BLS backup source", True
            elif ind_id == "NFP_3MA":
                return bls.get_nfp_3ma(), "BLS backup source", True
            raise ValueError(f"No BLS fetch logic for {ind_id}")
        elif source == "cboe":
            cboe = _get_cboe()
            value, is_proxy = cboe.get_put_call_ratio()
            note = "VIX/20 proxy — CBOE scrape failed" if is_proxy else ""
            return value, note, is_proxy
        elif source == "atlanta_fed":
            value, _ = _get_atlanta().get_gdpnow()
            return value, "", False
        elif source == "aaii":
            value, is_proxy = _get_aaii().get_bull_bear_spread()
            note = "AAII unavailable; neutral (0.0) returned" if is_proxy else ""
            return value, note, is_proxy
        elif source == "gpr":
            value, _ = _get_gpr().get_gpr()
            return value, "", False
        elif source == "manual":
            raise ValueError(f"{ind_id} requires manual/paid data input")
        else:
            raise ValueError(f"Unknown source '{source}' for {ind_id}")

    try:
        result = cached(cache_key, cadence, _do_fetch)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError(f"Unexpected cached result format for {ind_id}: {result!r}")
        value, proxy_note, is_proxy = result

        # Skip zero-weight indicators (present for context only, e.g. raw WTI)
        if cfg.get("weight", 0) == 0 and not cfg.get("optional", False):
            logger.debug(f"Skipping zero-weight indicator {ind_id}")
            return None

        iv = IndicatorValue(
            id=ind_id,
            value=float(value),
            unit=str(cfg.get("unit", "")),
            axis=cfg.get("axis", "RISK"),
            polarity=cfg.get("polarity", "ambiguous"),
            weight=float(cfg.get("weight", 0.0)),
            source=source,
            is_proxy=is_proxy,
            proxy_note=proxy_note,
            as_of=datetime.utcnow(),
        )
        if cfg.get("notes") and not proxy_note:
            iv = iv.model_copy(update={"proxy_note": cfg["notes"], "is_proxy": True})

        level = "DEBUG" if not is_proxy else "INFO"
        logger.log(
            level,
            f"[{source.upper():12}] {ind_id:20} = {value:>10.4f} {cfg.get('unit', '')}"
            + (f"  [PROXY: {proxy_note}]" if is_proxy else ""),
        )
        return iv

    except Exception as e:
        if is_optional:
            logger.debug(f"Optional indicator {ind_id} skipped: {e}")
        else:
            logger.warning(f"Failed to fetch {ind_id} ({source}): {e}")
        return None


# ── Main Fetcher class ─────────────────────────────────────────────────────────

class Fetcher:
    """Orchestrates fetching of all configured indicators into an IndicatorSnapshot."""

    def __init__(self) -> None:
        self.config = _load_config()
        self.settings = _load_settings()

    def fetch_all(self, indicator_ids: Optional[list[str]] = None) -> IndicatorSnapshot:
        """
        Fetch all (or a subset of) indicators and return a unified snapshot.

        Args:
            indicator_ids: If provided, only fetch these indicator IDs.
                           If None, fetch all configured indicators.

        Returns:
            IndicatorSnapshot with populated indicators, missing list, and warnings.
        """
        snapshot = IndicatorSnapshot()
        ids_to_fetch = indicator_ids or list(self.config.keys())

        logger.info(f"Fetching {len(ids_to_fetch)} indicators...")

        for ind_id in ids_to_fetch:
            if ind_id not in self.config:
                logger.warning(f"Indicator {ind_id} not in config — skipping")
                snapshot.missing.append(ind_id)
                continue

            cfg = self.config[ind_id]
            result = _fetch_indicator(ind_id, cfg)

            if result is not None:
                snapshot.indicators[ind_id] = result
                if result.is_proxy:
                    snapshot.warnings.append(
                        f"{ind_id}: proxy data used — {result.proxy_note}"
                    )
            else:
                snapshot.missing.append(ind_id)

        logger.info(
            f"Fetch complete: {len(snapshot.indicators)} fetched, "
            f"{len(snapshot.missing)} missing, "
            f"{len(snapshot.warnings)} warnings"
        )

        if snapshot.missing:
            logger.warning(f"Missing indicators: {snapshot.missing}")

        return snapshot

    def fetch_one(self, ind_id: str) -> Optional[IndicatorValue]:
        """
        Fetch a single indicator by ID.

        Args:
            ind_id: Indicator ID from indicators.yaml.

        Returns:
            IndicatorValue on success, None on failure.
        """
        if ind_id not in self.config:
            raise KeyError(f"Indicator '{ind_id}' not found in indicators.yaml")
        return _fetch_indicator(ind_id, self.config[ind_id])

    def get_spy_spot(self) -> float:
        """
        Return the current SPY price for trade construction.

        Returns:
            Current SPY close price.
        """
        return cached(
            "spot::SPY",
            "realtime",
            lambda: _get_yahoo().get_current_price("SPY"),
        )

    def get_risk_free_rate(self) -> float:
        """
        Return the current 2-year Treasury yield as risk-free rate.

        Returns:
            2Y yield as decimal (e.g. 0.045 for 4.5%).
        """
        try:
            rate_pct = cached(
                "indicator::UST_2Y",
                "daily",
                lambda: (_get_fred().get_latest("DGS2"), "", False),
            )
            if isinstance(rate_pct, tuple):
                rate_pct = rate_pct[0]
            return float(rate_pct) / 100.0
        except Exception as e:
            logger.warning(f"Could not fetch 2Y yield for risk-free rate: {e}")
            settings_fallback = self.settings.get("risk_free", {}).get("fallback", 0.05)
            return float(settings_fallback)

    def get_vix(self) -> float:
        """
        Return the current VIX level.

        Returns:
            VIX index value.
        """
        return cached(
            "indicator::VIX",
            "realtime",
            lambda: (_get_yahoo().get_current_price("^VIX"), "", False),
        )[0]
