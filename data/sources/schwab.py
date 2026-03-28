"""Schwab API client for live SPY option chains and account data.

Used for the LIVE system only (not backtest). Fetches current option chains
with Greeks for live trade construction, plus account/position data.

OAuth2 flow:
  - Access tokens expire every 30 minutes
  - Refresh tokens last 7 days
  - This client handles automatic token refresh

Requires:
  - SCHWAB_APP_KEY and SCHWAB_APP_SECRET env vars
  - Initial OAuth2 authorization via browser flow (one-time setup)

Usage:
    from data.sources.schwab import SchwabSource
    schwab = SchwabSource()
    chain = schwab.get_option_chain("SPY", dte_range=(21, 45))
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger

TOKEN_FILE = Path.home() / ".schwab_tokens.json"
AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
BASE_URL = "https://api.schwabapi.com/marketdata/v1"
TRADER_URL = "https://api.schwabapi.com/trader/v1"


class SchwabSource:
    """Schwab API client with automatic OAuth2 token management."""

    def __init__(self) -> None:
        self.app_key = os.environ.get("SCHWAB_APP_KEY", "")
        self.app_secret = os.environ.get("SCHWAB_APP_SECRET", "")
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._token_expiry: float = 0.0

        if self.app_key and self.app_secret:
            self._load_tokens()
        else:
            logger.warning(
                "SCHWAB_APP_KEY / SCHWAB_APP_SECRET not set. "
                "Schwab API calls will fail. Set up at developer.schwab.com"
            )

    # ── Token management ──────────────────────────────────────────────────────

    def _load_tokens(self) -> None:
        """Load saved tokens from disk."""
        if TOKEN_FILE.exists():
            with open(TOKEN_FILE) as f:
                data = json.load(f)
            self._access_token = data.get("access_token", "")
            self._refresh_token = data.get("refresh_token", "")
            self._token_expiry = data.get("expiry", 0.0)
            logger.debug("Loaded Schwab tokens from disk")

    def _save_tokens(self) -> None:
        """Save tokens to disk."""
        with open(TOKEN_FILE, "w") as f:
            json.dump({
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expiry": self._token_expiry,
            }, f)

    def _ensure_valid_token(self) -> None:
        """Refresh the access token if expired."""
        if time.time() < self._token_expiry - 60:
            return  # still valid with 60s buffer

        if not self._refresh_token:
            raise RuntimeError(
                "No refresh token available. Run initial OAuth2 authorization first. "
                "See: https://developer.schwab.com/products/trader-api--individual"
            )

        logger.info("Refreshing Schwab access token")
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            },
            auth=(self.app_key, self.app_secret),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        self._token_expiry = time.time() + data.get("expires_in", 1800)
        self._save_tokens()
        logger.info("Schwab token refreshed successfully")

    def _headers(self) -> dict[str, str]:
        """Return authorization headers."""
        self._ensure_valid_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    # ── Market data ───────────────────────────────────────────────────────────

    def get_option_chain(
        self,
        symbol: str = "SPY",
        dte_range: tuple[int, int] = (14, 75),
        strike_count: int = 40,
    ) -> dict[str, Any]:
        """Fetch the live option chain with Greeks.

        Args:
            symbol: Underlying symbol.
            dte_range: Min and max DTE to fetch.
            strike_count: Number of strikes around ATM.

        Returns:
            Raw API response dict with callExpDateMap and putExpDateMap.
        """
        from_date = (datetime.now() + timedelta(days=dte_range[0])).strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=dte_range[1])).strftime("%Y-%m-%d")

        params = {
            "symbol": symbol,
            "contractType": "ALL",
            "strikeCount": strike_count,
            "includeUnderlyingQuote": "TRUE",
            "strategy": "ANALYTICAL",
            "fromDate": from_date,
            "toDate": to_date,
        }

        resp = requests.get(
            f"{BASE_URL}/chains",
            headers=self._headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        logger.info(
            f"Schwab chain for {symbol}: "
            f"{len(data.get('callExpDateMap', {}))} call expiries, "
            f"{len(data.get('putExpDateMap', {}))} put expiries"
        )
        return data

    def get_quote(self, symbol: str = "SPY") -> dict[str, Any]:
        """Fetch a real-time quote.

        Args:
            symbol: Ticker symbol.

        Returns:
            Quote data dict.
        """
        resp = requests.get(
            f"{BASE_URL}/quotes/{symbol}",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Account data ──────────────────────────────────────────────────────────

    def get_accounts(self) -> list[dict]:
        """Fetch all linked accounts."""
        resp = requests.get(
            f"{TRADER_URL}/accounts",
            headers=self._headers(),
            params={"fields": "positions"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_positions(self, account_hash: str) -> list[dict]:
        """Fetch positions for a specific account.

        Args:
            account_hash: Encrypted account hash from get_accounts().

        Returns:
            List of position dicts.
        """
        resp = requests.get(
            f"{TRADER_URL}/accounts/{account_hash}",
            headers=self._headers(),
            params={"fields": "positions"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("securitiesAccount", {}).get("positions", [])

    # ── OAuth2 initial setup ──────────────────────────────────────────────────

    def get_auth_url(self, redirect_uri: str = "https://127.0.0.1") -> str:
        """Generate the OAuth2 authorization URL for initial setup.

        Args:
            redirect_uri: Redirect URI registered with Schwab developer portal.

        Returns:
            URL to open in browser for authorization.
        """
        return (
            f"{AUTH_URL}?"
            f"client_id={self.app_key}&"
            f"redirect_uri={redirect_uri}&"
            f"response_type=code"
        )

    def exchange_code(self, auth_code: str, redirect_uri: str = "https://127.0.0.1") -> None:
        """Exchange authorization code for tokens (one-time setup).

        Args:
            auth_code: Authorization code from OAuth2 callback.
            redirect_uri: Same redirect URI used in get_auth_url().
        """
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": redirect_uri,
            },
            auth=(self.app_key, self.app_secret),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        self._refresh_token = data["refresh_token"]
        self._token_expiry = time.time() + data.get("expires_in", 1800)
        self._save_tokens()
        logger.info("Schwab OAuth2 setup complete — tokens saved")
