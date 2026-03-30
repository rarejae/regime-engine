"""Per-asset 2-state HMMs: one independent HMM per risky ETF."""

import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ASSETS_TO_FIT = ["IVV", "QQQ", "VGLT", "IAU", "DBC"]
ETF_TICKERS = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}


def fetch_daily_prices() -> pd.DataFrame:
    """Fetch daily close prices for all 5 ETFs."""
    import yfinance as yf
    frames = {}
    for our, ticker in ETF_TICKERS.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our] = p
    return pd.DataFrame(frames).sort_index()


def compute_asset_features(prices: pd.Series) -> pd.DataFrame:
    """Compute 2 features for one asset: 21-day return + 21-day realized vol."""
    ret_21d = prices / prices.shift(21) - 1
    daily_ret = np.log(prices / prices.shift(1))
    vol_21d = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    return pd.DataFrame({"ret_21d": ret_21d, "vol_21d": vol_21d}, index=prices.index)


def fit_all_asset_hmms(
    prices_df: pd.DataFrame,
    window: int = 1260,
    refit_freq: int = 21,
) -> dict[str, pd.DataFrame]:
    """Fit independent 2-state HMM per asset. Returns dict of {asset: predictions_df}."""
    results = {}

    for asset in ASSETS_TO_FIT:
        if asset not in prices_df.columns:
            logger.warning(f"  {asset}: no price data, skipping")
            continue

        features = compute_asset_features(prices_df[asset])
        clean = features.dropna()

        if len(clean) < window:
            logger.warning(f"  {asset}: only {len(clean)} clean days (need {window})")
            continue

        preds = []
        last_fit = -refit_freq
        model = None
        scaler = None

        for i in range(window, len(clean)):
            if (i - last_fit) >= refit_freq or model is None:
                train = clean.iloc[max(0, i - window):i].values
                scaler = StandardScaler()
                train_s = scaler.fit_transform(train)
                try:
                    model = hmm.GaussianHMM(n_components=2, covariance_type="diag",
                                            n_iter=100, tol=1e-4, random_state=42)
                    model.fit(train_s)
                    last_fit = i
                except Exception:
                    continue

            if model is None or scaler is None:
                continue

            obs = clean.iloc[i:i + 1].values
            obs_s = scaler.transform(obs)
            try:
                probs = model.predict_proba(obs_s)[0]
                means = model.means_[:, 0]  # ret_21d
                bull_state = int(np.argmax(means))
                preds.append({
                    "date": clean.index[i],
                    "bull_prob": probs[bull_state],
                    "state": "bull" if probs[bull_state] > 0.5 else "bear",
                })
            except Exception:
                continue

        if preds:
            df = pd.DataFrame(preds).set_index("date")
            results[asset] = df
            n_trans = (df["state"] != df["state"].shift(1)).sum()
            n_years = len(df) / 252
            bull_pct = (df["state"] == "bull").mean()
            logger.info(f"  {asset}: {len(df)} days, {bull_pct:.0%} bull, "
                       f"{n_trans/n_years:.1f} transitions/yr")
        else:
            logger.warning(f"  {asset}: no predictions generated")

    return results


def get_asset_zones(
    hmm_predictions: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    persistence: int = 5,
) -> dict[str, str]:
    """Get HMM zone for each asset on a given day.

    Returns dict of {asset: "trending"/"not_trending"/"neutral"}.
    """
    zones = {}
    for asset in ASSETS_TO_FIT:
        pred = hmm_predictions.get(asset)
        if pred is None or day not in pred.index:
            zones[asset] = "neutral"
            continue

        bp = pred.loc[day, "bull_prob"]

        # Check persistence: look at last `persistence` days
        idx = pred.index.get_loc(day)
        if idx < persistence:
            zones[asset] = "neutral"
            continue

        recent = pred.iloc[idx - persistence + 1:idx + 1]["bull_prob"]

        if (recent > 0.7).all():
            zones[asset] = "trending"
        elif (recent < 0.3).all():
            zones[asset] = "not_trending"
        else:
            zones[asset] = "neutral"

    return zones


def get_hmm_diagnostics(hmm_predictions: dict[str, pd.DataFrame]) -> dict:
    """Compute diagnostics for each per-asset HMM."""
    diag = {}
    for asset, pred in hmm_predictions.items():
        if len(pred) < 252:
            continue
        bull_mask = pred["state"] == "bull"
        bear_mask = ~bull_mask
        n_trans = (pred["state"] != pred["state"].shift(1)).sum()
        n_years = len(pred) / 252

        # Run lengths
        runs = []
        cur = 1
        for i in range(1, len(pred)):
            if pred["state"].iloc[i] == pred["state"].iloc[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)

        diag[asset] = {
            "n_days": len(pred),
            "bull_pct": bull_mask.mean(),
            "bear_pct": bear_mask.mean(),
            "transitions": n_trans,
            "transitions_per_year": n_trans / n_years,
            "avg_run_length": np.mean(runs),
            "noisy": n_trans / n_years > 12,
        }
    return diag
