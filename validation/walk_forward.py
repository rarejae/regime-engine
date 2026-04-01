"""Walk-forward out-of-sample validation.

Split-sample and expanding-window tests to detect overfitting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from validation.backtest_harness import (
    BacktestConfig, BacktestResult, run_backtest, compute_metrics, _load_data
)


@dataclass
class SplitSampleResult:
    """Result from a single split-sample test."""
    split_name: str
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    sharpe_decay: float  # OOS Sharpe / IS Sharpe - 1
    return_decay: float  # OOS return / IS return - 1


@dataclass
class WalkForwardResult:
    """Result from expanding-window walk-forward test."""
    windows: list  # list of (train_end, test_start, test_end, BacktestResult)
    oos_sharpes: list
    oos_returns: list
    mean_oos_sharpe: float
    std_oos_sharpe: float
    min_oos_sharpe: float
    pct_positive_sharpe: float
    combined_oos_returns: pd.Series


def split_sample_test(config: BacktestConfig = None,
                       split_date: str = "2013-01-01",
                       data: dict = None) -> SplitSampleResult:
    """Run backtest on in-sample and out-of-sample halves.

    Default split at 2013-01-01 gives roughly equal halves of 2002-2025.
    """
    if config is None:
        config = BacktestConfig()
    if data is None:
        data = _load_data()

    # In-sample: bt_start to split_date
    cfg_is = BacktestConfig(**{
        **config.__dict__,
        "bt_end": split_date,
    })
    result_is = run_backtest(cfg_is, data)

    # Out-of-sample: split_date to end
    cfg_oos = BacktestConfig(**{
        **config.__dict__,
        "bt_start": split_date,
    })
    result_oos = run_backtest(cfg_oos, data)

    sharpe_decay = (result_oos.sharpe / result_is.sharpe - 1) if result_is.sharpe > 0 else float("nan")
    return_decay = (result_oos.ann_return / result_is.ann_return - 1) if result_is.ann_return > 0 else float("nan")

    return SplitSampleResult(
        split_name=f"IS(2002-{split_date[:4]}) / OOS({split_date[:4]}-end)",
        in_sample=result_is,
        out_of_sample=result_oos,
        sharpe_decay=sharpe_decay,
        return_decay=return_decay,
    )


def multi_split_test(config: BacktestConfig = None,
                      data: dict = None) -> list[SplitSampleResult]:
    """Run split-sample at multiple cut points to test robustness.

    Splits: 2010, 2013, 2016, 2019.
    """
    if data is None:
        data = _load_data()
    if config is None:
        config = BacktestConfig()

    splits = ["2010-01-01", "2013-01-01", "2016-01-01", "2019-01-01"]
    results = []
    for split in splits:
        r = split_sample_test(config, split, data)
        results.append(r)
    return results


def expanding_window_walkforward(config: BacktestConfig = None,
                                  data: dict = None,
                                  min_train_years: int = 5,
                                  test_years: int = 3,
                                  step_years: int = 2) -> WalkForwardResult:
    """Expanding-window walk-forward validation.

    Trains on expanding in-sample window, tests on fixed out-of-sample
    window. Steps forward by step_years each iteration.

    Default: min 5yr train, 3yr test, 2yr step.
    2002-2007 train → 2007-2010 test
    2002-2009 train → 2009-2012 test
    2002-2011 train → 2011-2014 test
    ... etc
    """
    if config is None:
        config = BacktestConfig()
    if data is None:
        data = _load_data()

    bt_start_year = int(config.bt_start[:4])
    # Find end of data
    daily_ret = data["daily_ret"]
    data_end_year = daily_ret.index.max().year

    windows = []
    oos_sharpes = []
    oos_returns = []
    oos_series_list = []

    train_end_year = bt_start_year + min_train_years

    while train_end_year + test_years <= data_end_year + 1:
        train_end = f"{train_end_year}-01-01"
        test_start = train_end
        test_end = f"{train_end_year + test_years}-01-01"

        # OOS test only (we don't re-optimize parameters, just test stability)
        cfg_oos = BacktestConfig(**{
            **config.__dict__,
            "bt_start": test_start,
            "bt_end": test_end,
        })
        result_oos = run_backtest(cfg_oos, data)

        windows.append((train_end, test_start, test_end, result_oos))
        oos_sharpes.append(result_oos.sharpe)
        oos_returns.append(result_oos.ann_return)
        oos_series_list.append(result_oos.daily_returns)

        train_end_year += step_years

    combined = pd.concat(oos_series_list).sort_index()
    # Remove duplicates (overlapping windows)
    combined = combined[~combined.index.duplicated(keep="first")]

    return WalkForwardResult(
        windows=windows,
        oos_sharpes=oos_sharpes,
        oos_returns=oos_returns,
        mean_oos_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0,
        std_oos_sharpe=float(np.std(oos_sharpes)) if oos_sharpes else 0,
        min_oos_sharpe=float(np.min(oos_sharpes)) if oos_sharpes else 0,
        pct_positive_sharpe=sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) if oos_sharpes else 0,
        combined_oos_returns=combined,
    )


def regime_stability_test(config: BacktestConfig = None,
                           data: dict = None) -> dict:
    """Test strategy performance across different market regimes.

    Evaluates Sharpe ratio during:
    - Bull markets (2003-2007, 2009-2019, 2020-2021)
    - Bear/crisis (2008-2009, 2020-Feb-Apr, 2022)
    - Rising rates (2016-2019, 2022-2023)
    - Low vol (2017)
    """
    if config is None:
        config = BacktestConfig()
    if data is None:
        data = _load_data()

    full = run_backtest(config, data)
    dr = full.daily_returns

    regimes = {
        "Bull 2003-2007": ("2003-01-01", "2007-10-01"),
        "GFC 2008-2009": ("2007-10-01", "2009-03-31"),
        "Recovery 2009-2019": ("2009-04-01", "2019-12-31"),
        "COVID Crash": ("2020-02-19", "2020-03-23"),
        "Post-COVID Bull": ("2020-04-01", "2021-12-31"),
        "2022 Bear": ("2022-01-01", "2022-10-31"),
        "2023-2024 Recovery": ("2023-01-01", "2024-12-31"),
    }

    results = {"full": full}
    for name, (start, end) in regimes.items():
        sub = dr[(dr.index >= pd.Timestamp(start)) & (dr.index <= pd.Timestamp(end))]
        if len(sub) > 20:
            met = compute_metrics(sub)
            results[name] = met
        else:
            results[name] = None

    return results


def run_all_walkforward(config: BacktestConfig = None,
                         data: dict = None) -> dict:
    """Run all walk-forward validation tests."""
    if data is None:
        data = _load_data()

    results = {}

    # Multi-split
    results["splits"] = multi_split_test(config, data)

    # Expanding window walk-forward
    results["walkforward"] = expanding_window_walkforward(config, data)

    # Regime stability
    results["regimes"] = regime_stability_test(config, data)

    # Pass/fail
    wf = results["walkforward"]
    results["all_oos_positive"] = wf.pct_positive_sharpe == 1.0
    results["min_oos_sharpe_above_zero"] = wf.min_oos_sharpe > 0
    results["mean_oos_sharpe_above_half"] = wf.mean_oos_sharpe > 0.5

    # Check split stability: no split should have >50% Sharpe *degradation*
    # Positive decay (OOS > IS) is fine; only negative decay is concerning
    worst_degradation = max((-s.sharpe_decay for s in results["splits"]
                             if not np.isnan(s.sharpe_decay)), default=0)
    results["max_split_decay"] = worst_degradation
    results["split_stable"] = worst_degradation < 0.50

    return results
