"""Greedy forward feature selection for HMM v3.

Builds all candidate z-scored features from expanded raw_indicators.parquet,
computes forward SPY returns, and selects features that maximize predictive
power subject to pairwise correlation cap.

Usage:
    PYTHONPATH=. python hmm/run_feature_selection.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from hmm.config import HMMConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET = Path("data/macro/raw_indicators.parquet")

# Map: feature_name → (raw_column, type)
# "direct" = ewm_zscore of raw column
# "derived" = needs custom computation
CANDIDATE_MAP = {
    # Existing v2 features
    "real_yield_z": ("REAL_YIELD_10Y", "direct"),
    "vix_z": ("VIX", "direct"),
    "copper_z": ("COPPER", "direct"),
    "housing_z": ("HOUSING_STARTS", "direct"),
    "skew_z": ("SKEW", "direct"),
    # Existing raw columns
    "nfci_z": ("NFCI", "direct"),
    "claims_z": ("CLAIMS_4WK", "direct"),
    "gold_z": ("GOLD", "direct"),
    "breakeven_z": ("BREAKEVEN_5Y", "direct"),
    "yield_curve_z": ("UST_2S10S", "direct"),
    "ffr_z": ("FFR_LEVEL", "direct"),
    "hy_oas_z": ("HY_OAS", "direct"),
    "hy_ig_diff_z": ("HY_IG_DIFF", "direct"),
    "fed_bs_z": ("FED_BS_DELTA_3M", "direct"),
    "copper_gold_z": ("COPPER_GOLD_RATIO", "direct"),
    "dxy_z": ("DXY", "direct"),
    "mort_30y_z": ("MORT_30Y", "direct"),
    "wti_z": ("WTI", "direct"),
    "vvix_z": ("VVIX", "direct"),
    "vix_term_z": ("VIX_TERM", "direct"),
    "ig_oas_z": ("IG_OAS", "direct"),
    "cpi_z": ("CPI_YOY", "direct"),
    "pce_z": ("PCE_CORE_YOY", "direct"),
    # New columns
    "ism_no_inv_z": ("ISM_NO_INV_SPREAD", "direct"),
    "stlfsi_z": ("STLFSI", "direct"),
    "m2_yoy_z": ("M2_YOY", "direct"),
    "ust_10y3m_z": ("UST_10Y_3M", "direct"),
    "qqq_spy_z": ("QQQ_SPY_RATIO", "direct"),
    "move_proxy_z": ("MOVE_PROXY", "direct"),
    # Derived price features
    "spy_rtn_21d": ("SPY_CLOSE", "spy_return"),
    "spy_vol_21d": ("SPY_CLOSE", "spy_vol"),
    "spx_rtn_3m_z": ("SPX_RTN_3M", "direct"),
}


def _ewm_zscore(series, halflife=252, min_periods=63, clip=3.0):
    mu = series.ewm(halflife=halflife, min_periods=min_periods).mean()
    sd = series.ewm(halflife=halflife, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - mu) / sd).clip(-clip, clip)


def build_all_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    """Build all candidate features from raw indicators."""
    result = pd.DataFrame(index=raw.index)

    for feat_name, (col, ftype) in CANDIDATE_MAP.items():
        if ftype == "direct":
            if col not in raw.columns:
                continue
            s = raw[col].astype(float)
            z = _ewm_zscore(s).shift(1)  # PIT safe
            result[feat_name] = z
        elif ftype == "spy_return":
            if col not in raw.columns:
                continue
            p = raw[col].astype(float)
            result[feat_name] = np.log(p / p.shift(21)).shift(1)
        elif ftype == "spy_vol":
            if col not in raw.columns:
                continue
            p = raw[col].astype(float)
            dr = np.log(p / p.shift(1))
            result[feat_name] = (dr.rolling(21, min_periods=10).std() * np.sqrt(252)).shift(1)

    return result


def greedy_forward_select(
    candidates: pd.DataFrame,
    forward_returns: pd.Series,
    max_features: int = 8,
    max_corr: float = 0.45,
    min_coverage: float = 0.85,
) -> list[tuple[str, float, float]]:
    """Greedy forward selection optimizing predictive power with correlation cap.

    Returns list of (feature_name, pred_power, worst_pair_corr) tuples.
    """
    # Filter by coverage
    valid_cols = []
    for col in candidates.columns:
        cov = candidates[col].notna().mean()
        if cov >= min_coverage:
            valid_cols.append(col)
        else:
            logger.debug(f"  {col}: dropped (coverage {cov:.0%} < {min_coverage:.0%})")

    logger.info(f"Candidates after coverage filter: {len(valid_cols)} of {len(candidates.columns)}")

    # Compute predictive power: |correlation| with forward returns
    aligned = candidates[valid_cols].copy()
    aligned["fwd"] = forward_returns
    aligned = aligned.dropna()

    pred_power = {}
    for col in valid_cols:
        r = aligned[col].corr(aligned["fwd"])
        pred_power[col] = abs(r) if not np.isnan(r) else 0

    selected: list[tuple[str, float, float]] = []
    remaining = set(valid_cols)

    for step in range(max_features):
        # Sort remaining by predictive power
        ranked = sorted(remaining, key=lambda c: pred_power[c], reverse=True)

        added = False
        for candidate in ranked:
            # Check pairwise correlation with all selected
            if selected:
                worst_corr = 0.0
                worst_pair = ""
                for sel_name, _, _ in selected:
                    pair_r = abs(aligned[candidate].corr(aligned[sel_name]))
                    if pair_r > worst_corr:
                        worst_corr = pair_r
                        worst_pair = sel_name

                if worst_corr >= max_corr:
                    logger.debug(f"  Step {step+1}: {candidate} rejected (|r|={worst_corr:.3f} with {worst_pair})")
                    continue
            else:
                worst_corr = 0.0

            selected.append((candidate, pred_power[candidate], worst_corr))
            remaining.remove(candidate)
            logger.info(f"  Step {step+1}: {candidate} — pred={pred_power[candidate]:.4f}, worst_pair_r={worst_corr:.3f}")
            added = True
            break

        if not added:
            logger.info(f"  Step {step+1}: no candidates pass correlation cap, stopping")
            break

    return selected


def main():
    raw = pd.read_parquet(PARQUET)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    logger.info(f"Raw data: {raw.shape[0]} rows x {raw.shape[1]} cols")

    # Build candidates
    candidates = build_all_candidates(raw)
    logger.info(f"Candidate features built: {candidates.shape[1]}")

    # Forward SPY returns (21-day)
    if "SPY_CLOSE" in raw.columns:
        spy = raw["SPY_CLOSE"].astype(float)
    else:
        logger.error("SPY_CLOSE not found")
        return

    fwd_21d = np.log(spy / spy.shift(-21))  # Forward return (positive = SPY goes up)
    fwd_21d = fwd_21d.reindex(candidates.index)

    logger.info(f"Forward returns coverage: {fwd_21d.notna().mean():.1%}")

    # Run selection
    logger.info(f"\n{'='*60}")
    logger.info("GREEDY FORWARD SELECTION (max |r| < 0.45)")
    logger.info(f"{'='*60}")

    selected = greedy_forward_select(candidates, fwd_21d, max_features=8, max_corr=0.45)

    logger.info(f"\n{'='*60}")
    logger.info("SELECTED FEATURES")
    logger.info(f"{'='*60}")

    feat_names = [s[0] for s in selected]
    for i, (name, pred, worst) in enumerate(selected):
        logger.info(f"  {i+1}. {name:20s}  pred_power={pred:.4f}  worst_pair_r={worst:.3f}")

    # Correlation matrix
    aligned = candidates[feat_names].dropna()
    corr = aligned.corr()
    max_r = 0
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            r = abs(corr.iloc[i, j])
            if r > max_r:
                max_r = r

    logger.info(f"\nMax pairwise |r|: {max_r:.3f}")
    logger.info(f"Mean pairwise |r|: {corr.abs().values[np.triu_indices(len(feat_names), k=1)].mean():.3f}")

    logger.info(f"\nCorrelation matrix:")
    header = "              " + "  ".join(f"{n[:10]:>10}" for n in feat_names)
    logger.info(header)
    for i, n1 in enumerate(feat_names):
        row = f"{n1[:14]:>14}"
        for j, n2 in enumerate(feat_names):
            row += f"  {corr.loc[n1, n2]:>10.3f}"
        logger.info(row)

    # Print config update
    raw_map = {}
    for f in feat_names:
        if f in CANDIDATE_MAP:
            col, ftype = CANDIDATE_MAP[f]
            raw_map[f] = col

    logger.info(f"\n{'='*60}")
    logger.info("CONFIG UPDATE (paste into hmm/config.py)")
    logger.info(f"{'='*60}")
    logger.info(f"hmm_features = {feat_names}")
    logger.info(f"raw_indicator_map = {raw_map}")


if __name__ == "__main__":
    main()
