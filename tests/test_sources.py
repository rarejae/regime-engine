"""Tests for data source modules.

Each test verifies that the source returns data in the expected format.
Tests requiring live API calls are marked with @pytest.mark.integration
and can be skipped in CI with: pytest -m "not integration"
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── FRED Tests ────────────────────────────────────────────────────────────────

class TestFREDSource:
    """Tests for data/sources/fred.py"""

    @pytest.fixture
    def fred_source(self):
        """Return a FREDSource with a mocked fredapi client."""
        from data.sources.fred import FREDSource
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            with patch("fredapi.Fred") as mock_fred_cls:
                source = FREDSource()
                source.client = mock_fred_cls.return_value
                yield source

    def test_init_raises_without_api_key(self):
        """FREDSource should raise if FRED_API_KEY is not set."""
        from data.sources.fred import FREDSource
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FRED_API_KEY", None)
            with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
                FREDSource()

    def test_get_latest_returns_float(self, fred_source):
        """get_latest should return the last non-NaN value as float."""
        import numpy as np
        mock_series = pd.Series([4.5, 4.75, 5.0, np.nan], index=pd.date_range("2024-01-01", periods=4, freq="D"))
        fred_source.client.get_series.return_value = mock_series
        result = fred_source.get_latest("DFF")
        assert result == 5.0
        assert isinstance(result, float)

    def test_get_yoy_pct_correct(self, fred_source):
        """get_yoy_pct should compute (current / year_ago - 1) * 100."""
        dates = pd.date_range("2023-01-01", periods=14, freq="ME")
        values = [100.0] * 13 + [104.0]  # 4% YoY
        mock_series = pd.Series(values, index=dates)
        fred_source.client.get_series.return_value = mock_series
        result = fred_source.get_yoy_pct("CPIAUCSL")
        assert abs(result - 4.0) < 0.01

    def test_get_spread_bps(self, fred_source):
        """get_spread_bps should return (a - b) * 100 in basis points."""
        mock_a = pd.Series([4.5], index=[pd.Timestamp("2024-01-01")])
        mock_b = pd.Series([4.0], index=[pd.Timestamp("2024-01-01")])
        fred_source.client.get_series.side_effect = [mock_a, mock_b]
        result = fred_source.get_spread_bps("DGS10", "DGS2")
        assert result == pytest.approx(50.0)

    def test_get_delta_12m_in_bps(self, fred_source):
        """get_delta_12m should return change * 100 (bps)."""
        dates = pd.date_range("2023-01-01", periods=400, freq="D")
        # Start at 3.0%, end at 5.0% → delta = +2.0 → +200 bps
        values = [3.0] * 365 + [5.0] * 35
        mock_series = pd.Series(values, index=dates)
        fred_source.client.get_series.return_value = mock_series
        result = fred_source.get_delta_12m("DFF")
        assert abs(result - 200.0) < 10  # within 10 bps due to day alignment

    def test_get_mom_3ma_thousands(self, fred_source):
        """get_mom_3ma_thousands should average the last 3 monthly changes."""
        # NFP levels: each month adds exactly 250K → 3-month avg = 250K
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        values = [158000, 158250, 158500, 158750, 159000]  # constant +250K each month
        mock_series = pd.Series(values, index=dates)
        fred_source.client.get_series.return_value = mock_series
        result = fred_source.get_mom_3ma_thousands("PAYEMS")
        assert abs(result - 250.0) < 1.0


# ── Yahoo Tests ───────────────────────────────────────────────────────────────

class TestYahooSource:
    """Tests for data/sources/yahoo.py"""

    @pytest.fixture
    def yahoo_source(self):
        from data.sources.yahoo import YahooSource
        return YahooSource()

    def test_get_return_3m_pct_calculation(self, yahoo_source):
        """get_return_3m_pct should compute trailing 63-bar return."""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            prices = [100.0] * 63 + [110.0]  # 10% gain over 63 bars
            idx = pd.date_range("2023-01-01", periods=64, freq="B")
            mock_hist = pd.DataFrame({"Close": prices}, index=idx)
            mock_ticker_cls.return_value.history.return_value = mock_hist
            result = yahoo_source.get_return_3m_pct("SPY")
            assert abs(result - 10.0) < 0.01

    def test_vix_term_structure_is_difference(self, yahoo_source):
        """VIX term structure should be VIX3M - VIX."""
        with patch.object(yahoo_source, "get_current_price") as mock_price:
            mock_price.side_effect = lambda t: 18.0 if t == "^VIX3M" else 20.0
            result = yahoo_source.get_vix_term_structure()
            assert result == pytest.approx(-2.0)

    def test_up_down_ratio_10d_returns_ratio(self, yahoo_source):
        """up_down_ratio_10d should return positive ratio."""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            # 7 up days, 3 down days
            prices = [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108]
            idx = pd.date_range("2024-01-01", periods=11, freq="B")
            mock_hist = pd.DataFrame({"Close": prices}, index=idx)
            mock_ticker_cls.return_value.history.return_value = mock_hist
            result = yahoo_source.get_up_down_ratio_10d("RSP")
            assert result > 0

    def test_filter_chain_removes_illiquid(self):
        """filter_chain should remove zero OI and wide bid-ask rows."""
        from utils.helpers import filter_chain
        chain = pd.DataFrame({
            "strike": [400, 405, 410],
            "bid": [0.0, 1.00, 0.50],
            "ask": [0.0, 1.05, 2.00],
            "openInterest": [0, 500, 200],
            "volume": [0, 100, 5],
            "impliedVolatility": [0.20, 0.18, 0.22],
        })
        result = filter_chain(chain, min_open_interest=100, max_bid_ask_spread_pct=0.20, min_volume=10)
        assert len(result) == 1
        assert result.iloc[0]["strike"] == 405


# ── BLS Tests ─────────────────────────────────────────────────────────────────

class TestBLSSource:
    """Tests for data/sources/bls.py"""

    @pytest.fixture
    def bls_source(self):
        from data.sources.bls import BLSSource
        return BLSSource()

    def test_parse_bls_data_sorts_ascending(self, bls_source):
        """BLS parser should return data sorted ascending by date."""
        raw = [
            {"year": "2024", "period": "M03", "value": "310.0"},
            {"year": "2024", "period": "M01", "value": "308.0"},
            {"year": "2024", "period": "M02", "value": "309.0"},
        ]
        df = bls_source._parse(raw)
        dates = list(df["date"])
        assert dates == sorted(dates)
        assert len(df) == 3

    def test_parse_bls_skips_annual_m13(self, bls_source):
        """BLS parser should skip M13 (annual) entries."""
        raw = [
            {"year": "2024", "period": "M01", "value": "308.0"},
            {"year": "2024", "period": "M13", "value": "309.0"},  # annual
        ]
        df = bls_source._parse(raw)
        assert len(df) == 1

    def test_get_series_makes_correct_request(self, bls_source):
        """get_series should POST to BLS API with correct payload."""
        mock_response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [{
                    "data": [{"year": "2024", "period": "M01", "value": "308.0"}]
                }]
            }
        }
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()
            df = bls_source.get_series("CUSR0000SA0", start_year=2024, end_year=2024)
        assert len(df) == 1
        call_kwargs = mock_post.call_args
        assert "CUSR0000SA0" in str(call_kwargs)


# ── CBOE Tests ────────────────────────────────────────────────────────────────

class TestCBOESource:
    """Tests for data/sources/cboe.py"""

    def test_put_call_returns_tuple(self):
        """get_put_call_ratio should always return (float, bool)."""
        from data.sources.cboe import CBOESource
        cboe = CBOESource()
        with patch.object(cboe, "_fetch_from_cboe_json") as mock_fetch:
            mock_fetch.return_value = 0.85
            ratio, is_proxy = cboe.get_put_call_ratio()
        assert isinstance(ratio, float)
        assert isinstance(is_proxy, bool)
        assert not is_proxy

    def test_fallback_to_vix_proxy(self):
        """Should fall back to VIX proxy if CBOE fetch fails."""
        from data.sources.cboe import CBOESource
        cboe = CBOESource()
        with patch.object(cboe, "_fetch_from_cboe_json", side_effect=Exception("scrape failed")):
            with patch("yfinance.Ticker") as mock_ticker:
                hist = pd.DataFrame({"Close": [20.0]}, index=[pd.Timestamp.utcnow()])
                mock_ticker.return_value.history.return_value = hist
                ratio, is_proxy = cboe.get_put_call_ratio()
        assert is_proxy
        assert ratio == pytest.approx(1.0)  # VIX=20 → 20/20 = 1.0


# ── GPR Tests ─────────────────────────────────────────────────────────────────

class TestGPRSource:
    """Tests for data/sources/gpr.py"""

    def test_parse_gpr_excel_handles_variant_columns(self):
        """GPR parser should handle different column naming conventions."""
        from data.sources.gpr import GPRSource
        import io

        # Simulate Excel with 'DATE' and 'GPR' columns
        df = pd.DataFrame({
            "DATE": pd.date_range("2024-01-01", periods=5, freq="D"),
            "GPR": [95.0, 98.0, 102.0, 99.0, 105.0],
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        buf.seek(0)

        result = GPRSource._parse_gpr_excel(buf)
        assert len(result) == 5
        assert list(result.columns) == ["date", "gpr"]
        assert result.iloc[-1]["gpr"] == 105.0


# ── AtlantaFed Tests ──────────────────────────────────────────────────────────

class TestAtlantaFedSource:
    """Tests for data/sources/atlanta_fed.py"""

    def test_html_fallback_parses_gdpnow_pattern(self):
        """HTML scraper should find GDPNow estimate from page text."""
        from data.sources.atlanta_fed import AtlantaFedSource
        from bs4 import BeautifulSoup

        source = AtlantaFedSource()
        html = """<html><body>
        <p>The GDPNow model estimate for real GDP growth is 2.4 percent for Q1 2025.</p>
        </body></html>"""

        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status = MagicMock()
            mock_get.return_value.text = html
            estimate, as_of = source._from_html()

        assert estimate == pytest.approx(2.4)


# ── AAII Tests ────────────────────────────────────────────────────────────────

class TestAAIISource:
    """Tests for data/sources/aaii.py"""

    def test_html_scraper_extracts_percentages(self):
        """AAII scraper should parse bullish/bearish/neutral from HTML."""
        from data.sources.aaii import AAIISource

        source = AAIISource()
        html = """<html><body>
        <p>Bullish 38.5%</p>
        <p>Bearish 28.1%</p>
        <p>Neutral 33.4%</p>
        </body></html>"""

        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status = MagicMock()
            mock_get.return_value.text = html
            result = source._from_html()

        assert result["bullish"] == pytest.approx(38.5)
        assert result["bearish"] == pytest.approx(28.1)
        assert result["bull_bear_spread"] == pytest.approx(10.4)

    def test_neutral_fallback_to_complement(self):
        """If neutral not on page, should compute 100 - bull - bear."""
        from data.sources.aaii import AAIISource
        source = AAIISource()
        html = """<html><body>
        <p>Bullish 40.0%</p>
        <p>Bearish 30.0%</p>
        </body></html>"""
        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status = MagicMock()
            mock_get.return_value.text = html
            result = source._from_html()
        assert result["neutral"] == pytest.approx(30.0)
