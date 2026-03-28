"""Cross-reference validation of Dubach data against optionsDX.

Compares overlapping years (2010–2023) between optionsDX and Dubach datasets
to verify Dubach data quality. If validation passes, the 2008–2009 Dubach
data is usable for backtesting. If it fails, only optionsDX data is used.

Validation steps:
  1. Load optionsDX and Dubach processed Parquet for overlapping years
  2. Match contracts by (date, expiry, strike, type)
  3. Compare: bid, ask, IV, delta — compute correlation and MAE
  4. Flag days/contracts where divergence exceeds thresholds
  5. Generate pass/fail per year

Usage:
    python -m data.validate_dubach
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

from utils.logging import setup_logging

PROCESSED_DIR = Path(__file__).parent / "processed"
VALIDATION_DIR = Path(__file__).parent / "validation"

# Thresholds for validation
DELTA_MAE_THRESHOLD = 0.05
IV_MAE_THRESHOLD = 0.02
BID_ASK_CORR_THRESHOLD = 0.90
MIN_MATCH_RATE = 0.50  # at least 50% of contracts must match


def _load_year_optionsdx(year: int) -> pd.DataFrame | None:
    """Load optionsDX processed Parquet for a given year."""
    path = PROCESSED_DIR / f"spy_options_{year}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_year_dubach(year: int) -> pd.DataFrame | None:
    """Load Dubach processed Parquet for a given year."""
    path = PROCESSED_DIR / f"spy_options_{year}_dubach.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def validate_year(year: int) -> dict:
    """Validate Dubach data against optionsDX for a single year.

    Args:
        year: Year to validate.

    Returns:
        Dict with validation metrics and pass/fail status.
    """
    odx = _load_year_optionsdx(year)
    dub = _load_year_dubach(year)

    if odx is None:
        return {"year": year, "status": "skip", "reason": "no optionsdx data"}
    if dub is None:
        return {"year": year, "status": "skip", "reason": "no dubach data"}

    logger.info(f"Validating {year}: optionsDX={len(odx):,} rows, Dubach={len(dub):,} rows")

    # Ensure date columns are proper types
    for df in [odx, dub]:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["expiry"] = pd.to_datetime(df["expiry"])

    # Join on (trade_date, expiry, strike, option_type)
    merge_keys = ["trade_date", "expiry", "strike", "option_type"]
    merged = odx.merge(
        dub,
        on=merge_keys,
        suffixes=("_odx", "_dub"),
        how="inner",
    )

    if len(merged) == 0:
        return {"year": year, "status": "fail", "reason": "no matching contracts", "match_count": 0}

    match_rate = len(merged) / min(len(odx), len(dub))
    logger.info(f"  Matched {len(merged):,} contracts (rate={match_rate:.2%})")

    results: dict = {
        "year": year,
        "odx_rows": len(odx),
        "dub_rows": len(dub),
        "matched": len(merged),
        "match_rate": round(match_rate, 4),
    }

    # Compare fields
    for field in ["bid", "ask", "iv", "delta"]:
        odx_col = f"{field}_odx"
        dub_col = f"{field}_dub"

        if odx_col not in merged.columns or dub_col not in merged.columns:
            results[f"{field}_status"] = "skip"
            continue

        valid = merged[[odx_col, dub_col]].dropna()
        if len(valid) < 100:
            results[f"{field}_status"] = "insufficient_data"
            continue

        mae = np.mean(np.abs(valid[odx_col] - valid[dub_col]))
        corr = valid[odx_col].corr(valid[dub_col])

        results[f"{field}_mae"] = round(float(mae), 6)
        results[f"{field}_corr"] = round(float(corr), 4) if not np.isnan(corr) else None

        logger.info(f"  {field}: MAE={mae:.6f}, corr={corr:.4f}")

    # Determine pass/fail
    delta_mae = results.get("delta_mae", float("inf"))
    iv_mae = results.get("iv_mae", float("inf"))
    bid_corr = results.get("bid_corr", 0)

    passed = (
        delta_mae <= DELTA_MAE_THRESHOLD
        and iv_mae <= IV_MAE_THRESHOLD
        and (bid_corr is not None and bid_corr >= BID_ASK_CORR_THRESHOLD)
        and match_rate >= MIN_MATCH_RATE
    )

    results["status"] = "pass" if passed else "fail"
    if not passed:
        reasons = []
        if delta_mae > DELTA_MAE_THRESHOLD:
            reasons.append(f"delta MAE {delta_mae:.4f} > {DELTA_MAE_THRESHOLD}")
        if iv_mae > IV_MAE_THRESHOLD:
            reasons.append(f"IV MAE {iv_mae:.4f} > {IV_MAE_THRESHOLD}")
        if bid_corr is not None and bid_corr < BID_ASK_CORR_THRESHOLD:
            reasons.append(f"bid corr {bid_corr:.4f} < {BID_ASK_CORR_THRESHOLD}")
        if match_rate < MIN_MATCH_RATE:
            reasons.append(f"match rate {match_rate:.2%} < {MIN_MATCH_RATE:.0%}")
        results["fail_reasons"] = reasons

    logger.info(f"  Year {year}: {'PASS' if passed else 'FAIL'}")
    return results


def validate_all() -> dict:
    """Run validation for all overlapping years (2010–2023).

    Returns:
        Full validation report with per-year results and overall verdict.
    """
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "thresholds": {
            "delta_mae": DELTA_MAE_THRESHOLD,
            "iv_mae": IV_MAE_THRESHOLD,
            "bid_corr": BID_ASK_CORR_THRESHOLD,
            "min_match_rate": MIN_MATCH_RATE,
        },
        "years": {},
        "overall": "unknown",
    }

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for year in range(2010, 2024):
        result = validate_year(year)
        report["years"][year] = result
        if result["status"] == "pass":
            pass_count += 1
        elif result["status"] == "fail":
            fail_count += 1
        else:
            skip_count += 1

    # Overall verdict: pass if majority of validated years pass
    validated = pass_count + fail_count
    if validated == 0:
        report["overall"] = "no_data"
        report["dubach_2008_2009_usable"] = False
    elif pass_count / validated >= 0.75:
        report["overall"] = "pass"
        report["dubach_2008_2009_usable"] = True
    else:
        report["overall"] = "fail"
        report["dubach_2008_2009_usable"] = False

    report["summary"] = {
        "years_passed": pass_count,
        "years_failed": fail_count,
        "years_skipped": skip_count,
    }

    logger.info(
        f"Validation complete: {pass_count} passed, {fail_count} failed, {skip_count} skipped. "
        f"Overall: {report['overall']}. "
        f"Dubach 2008-2009 usable: {report['dubach_2008_2009_usable']}"
    )

    return report


def save_report(report: dict) -> Path:
    """Save validation report to JSON file.

    Args:
        report: Validation report dict.

    Returns:
        Path to saved report.
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VALIDATION_DIR / "dubach_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {out_path}")
    return out_path


def is_dubach_validated() -> bool:
    """Check if Dubach data has passed validation.

    Returns:
        True if the most recent validation report shows a pass.
    """
    report_path = VALIDATION_DIR / "dubach_validation_report.json"
    if not report_path.exists():
        return False
    with open(report_path) as f:
        report = json.load(f)
    return report.get("dubach_2008_2009_usable", False)


@click.command()
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def main(debug: bool) -> None:
    """Validate Dubach data against optionsDX for overlapping years."""
    setup_logging("DEBUG" if debug else None)
    report = validate_all()
    save_report(report)


if __name__ == "__main__":
    main()
