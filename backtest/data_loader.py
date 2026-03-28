"""DuckDB-based data loader for backtest option chain queries.

Queries Parquet files directly via DuckDB — no import/ETL step needed.
DuckDB reads Parquet natively with zero network latency.

Usage:
    from backtest.data_loader import BacktestDataLoader
    loader = BacktestDataLoader()
    chain = loader.get_chain(trade_date="2020-03-15", dte_range=(21, 45))
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from loguru import logger

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class BacktestDataLoader:
    """DuckDB wrapper for querying historical option chain Parquet files."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Args:
            data_dir: Path to processed Parquet directory. Defaults to data/processed/.
        """
        self.data_dir = data_dir or PROCESSED_DIR
        self.con = duckdb.connect(":memory:")
        self._available_years: list[int] | None = None

    def available_years(self, source: str | None = None) -> list[int]:
        """Return sorted list of years with available data.

        Args:
            source: Filter by source ('optionsdx', 'dubach', or None for all).
        """
        if self._available_years is None:
            files = list(self.data_dir.glob("spy_options_*.parquet"))
            years = set()
            for f in files:
                name = f.stem  # spy_options_2020 or spy_options_2020_dubach
                parts = name.replace("spy_options_", "").split("_")
                try:
                    years.add(int(parts[0]))
                except ValueError:
                    continue
            self._available_years = sorted(years)
        return self._available_years

    def _parquet_glob(self, year: int | None = None, source: str | None = None) -> str:
        """Build a glob pattern for Parquet files.

        Args:
            year: Specific year, or None for all years.
            source: 'optionsdx' (no suffix), 'dubach' (_dubach suffix), or None (all).
        """
        if year is not None and source == "dubach":
            return str(self.data_dir / f"spy_options_{year}_dubach.parquet")
        elif year is not None and source == "optionsdx":
            return str(self.data_dir / f"spy_options_{year}.parquet")
        elif year is not None:
            return str(self.data_dir / f"spy_options_{year}*.parquet")
        elif source == "optionsdx":
            # Only non-dubach files
            return str(self.data_dir / "spy_options_[0-9][0-9][0-9][0-9].parquet")
        else:
            return str(self.data_dir / "spy_options_*.parquet")

    def get_chain(
        self,
        trade_date: date | str,
        dte_range: tuple[int, int] = (14, 75),
        option_type: str | None = None,
        delta_range: tuple[float, float] | None = None,
        min_oi: int = 100,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Query the option chain for a specific trading day.

        Args:
            trade_date: Trading day to query.
            dte_range: (min_dte, max_dte) filter.
            option_type: 'C', 'P', or None for both.
            delta_range: Filter by absolute delta range.
            min_oi: Minimum open interest.
            source: Filter by data source.

        Returns:
            DataFrame with matching option contracts.
        """
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        year = trade_date.year
        parquet_glob = self._parquet_glob(year, source)

        where = [
            f"trade_date = '{trade_date.isoformat()}'",
            f"dte BETWEEN {dte_range[0]} AND {dte_range[1]}",
            f"open_interest >= {min_oi}",
        ]

        if option_type:
            where.append(f"option_type = '{option_type}'")
        if delta_range:
            where.append(f"ABS(delta) BETWEEN {delta_range[0]} AND {delta_range[1]}")

        query = f"""
            SELECT *
            FROM '{parquet_glob}'
            WHERE {' AND '.join(where)}
            ORDER BY strike ASC
        """

        try:
            result = self.con.execute(query).fetchdf()
            logger.debug(f"Chain query for {trade_date}: {len(result)} rows")
            return result
        except Exception as e:
            logger.warning(f"Chain query failed for {trade_date}: {e}")
            return pd.DataFrame()

    def get_spread_value(
        self,
        trade_date: date | str,
        short_strike: float,
        long_strike: float,
        expiry: date | str,
        option_type: str,
        source: str | None = None,
    ) -> Optional[float]:
        """Get the current mid-price value of a spread on a given date.

        Args:
            trade_date: Observation date.
            short_strike: Short leg strike.
            long_strike: Long leg strike.
            expiry: Option expiration date.
            option_type: 'C' or 'P'.
            source: Data source filter.

        Returns:
            Spread mid-price value, or None if data unavailable.
        """
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)
        if isinstance(expiry, str):
            expiry = date.fromisoformat(expiry)

        year = trade_date.year
        parquet_glob = self._parquet_glob(year, source)

        query = f"""
            SELECT strike, mid, bid, ask
            FROM '{parquet_glob}'
            WHERE trade_date = '{trade_date.isoformat()}'
              AND expiry = '{expiry.isoformat()}'
              AND option_type = '{option_type}'
              AND strike IN ({short_strike}, {long_strike})
        """

        try:
            result = self.con.execute(query).fetchdf()
        except Exception:
            return None

        if len(result) < 2:
            return None

        short_row = result[result["strike"] == short_strike]
        long_row = result[result["strike"] == long_strike]

        if short_row.empty or long_row.empty:
            return None

        short_mid = float(short_row.iloc[0]["mid"])
        long_mid = float(long_row.iloc[0]["mid"])

        # For credit spreads: value = short_mid - long_mid
        return short_mid - long_mid

    def get_spy_close(self, trade_date: date | str, source: str | None = None) -> Optional[float]:
        """Get SPY closing price for a given date.

        Args:
            trade_date: Trading day.
            source: Data source filter.

        Returns:
            SPY close price, or None if unavailable.
        """
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        year = trade_date.year
        parquet_glob = self._parquet_glob(year, source)

        query = f"""
            SELECT underlying_close
            FROM '{parquet_glob}'
            WHERE trade_date = '{trade_date.isoformat()}'
              AND underlying_close > 0
            LIMIT 1
        """

        try:
            result = self.con.execute(query).fetchdf()
            if result.empty:
                return None
            return float(result.iloc[0]["underlying_close"])
        except Exception:
            return None

    def trading_days(
        self,
        start_date: date | str,
        end_date: date | str,
        source: str | None = None,
    ) -> list[date]:
        """Get all trading days with data in a date range.

        Args:
            start_date: Start of range.
            end_date: End of range.
            source: Data source filter.

        Returns:
            Sorted list of trading dates.
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        parquet_glob = self._parquet_glob(source=source)

        query = f"""
            SELECT DISTINCT trade_date
            FROM '{parquet_glob}'
            WHERE trade_date BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
            ORDER BY trade_date
        """

        try:
            result = self.con.execute(query).fetchdf()
            return [d.date() if hasattr(d, 'date') else d for d in result["trade_date"].tolist()]
        except Exception as e:
            logger.warning(f"Trading days query failed: {e}")
            return []
