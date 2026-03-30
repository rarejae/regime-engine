"""2-state HMM trend detector on SPY returns, realized vol, and credit spread."""

import logging
import os

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fetch_daily_data() -> pd.DataFrame:
    """Fetch SPY prices and HY credit spread."""
    import yfinance as yf

    frames = {}

    # SPY daily close
    spy = yf.download("SPY", start="1993-01-01", progress=False)
    if spy is not None and not spy.empty:
        p = spy["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        frames["spy_close"] = p

    # HY credit spread from FRED
    key = os.environ.get("FRED_API_KEY")
    if key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=key)
            hy = fred.get_series("BAMLH0A0HYM2", observation_start="1997-01-01")
            if hy is not None:
                hy.index = pd.to_datetime(hy.index)
                frames["hy_spread"] = hy
        except Exception as e:
            logger.warning(f"FRED HY spread failed: {e}")

    # Fallback: HYG ETF inverse as spread proxy
    if "hy_spread" not in frames:
        hyg = yf.download("HYG", start="2007-04-01", progress=False)
        if hyg is not None and not hyg.empty:
            hp = hyg["Close"]
            if hasattr(hp, "columns"): hp = hp.iloc[:, 0]
            hp.index = pd.to_datetime(hp.index).tz_localize(None)
            # Inverse: falling HYG = widening spread
            frames["hy_spread"] = -hp.pct_change().rolling(21).sum() * 100

    df = pd.DataFrame(frames).sort_index()
    logger.info(f"Daily data: {df.shape[0]} days, cols={list(df.columns)}")
    return df


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute 3 HMM features from raw data."""
    spy = raw["spy_close"]

    # 1. 21-day rolling return
    ret_21d = spy / spy.shift(21) - 1

    # 2. 21-day realized volatility
    daily_ret = np.log(spy / spy.shift(1))
    vol_21d = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)

    # 3. HY credit spread 21-day change
    if "hy_spread" in raw.columns:
        hy_chg_21d = raw["hy_spread"] - raw["hy_spread"].shift(21)
    else:
        hy_chg_21d = pd.Series(0, index=raw.index)

    features = pd.DataFrame({
        "ret_21d": ret_21d,
        "vol_21d": vol_21d,
        "hy_chg_21d": hy_chg_21d,
    }, index=raw.index)

    return features


def compute_features_2f(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute 2-feature version (no credit spread)."""
    spy = raw["spy_close"]
    ret_21d = spy / spy.shift(21) - 1
    daily_ret = np.log(spy / spy.shift(1))
    vol_21d = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)

    return pd.DataFrame({
        "ret_21d": ret_21d,
        "vol_21d": vol_21d,
    }, index=raw.index)


def fit_and_predict_rolling(
    features: pd.DataFrame,
    window: int = 1260,
    refit_freq: int = 21,
    n_states: int = 2,
) -> pd.DataFrame:
    """Rolling HMM fit with forward-only filtered probabilities.

    Returns DataFrame with columns: bull_prob, bear_prob, state, spy_close.
    """
    clean = features.dropna()
    feat_cols = list(clean.columns)

    results = []
    last_fit = -refit_freq
    model = None
    scaler = None

    for i in range(window, len(clean)):
        # Refit monthly
        if (i - last_fit) >= refit_freq or model is None:
            train = clean.iloc[max(0, i - window):i][feat_cols].values
            if len(train) < window // 2:
                continue

            scaler = StandardScaler()
            train_s = scaler.fit_transform(train)

            try:
                model = hmm.GaussianHMM(
                    n_components=n_states, covariance_type="diag",
                    n_iter=100, tol=1e-4, random_state=42,
                )
                model.fit(train_s)
                last_fit = i
            except Exception:
                continue

        if model is None or scaler is None:
            continue

        # Predict on single observation (filtered, not smoothed)
        obs = clean.iloc[i:i+1][feat_cols].values
        obs_s = scaler.transform(obs)

        try:
            probs = model.predict_proba(obs_s)[0]

            # Label: state with higher mean return = bull
            means = model.means_[:, 0]  # first feature = ret_21d
            bull_state = int(np.argmax(means))
            bear_state = 1 - bull_state

            results.append({
                "date": clean.index[i],
                "bull_prob": probs[bull_state],
                "bear_prob": probs[bear_state],
                "state": "bull" if probs[bull_state] > 0.5 else "bear",
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
    return df


def apply_persistence_filter(states: pd.Series, threshold: int = 5) -> pd.Series:
    """Only transition after `threshold` consecutive days in new state."""
    filtered = states.copy()
    prev_confirmed = states.iloc[0] if len(states) > 0 else "bull"
    consec = 0
    current = prev_confirmed

    for i in range(len(states)):
        raw = states.iloc[i]
        if raw == current:
            consec += 1
        else:
            consec = 1
            current = raw

        if consec >= threshold:
            prev_confirmed = current

        filtered.iloc[i] = prev_confirmed

    return filtered
