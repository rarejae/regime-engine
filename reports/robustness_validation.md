# TAA Robustness Validation Report

Generated: 2026-03-31 17:24

## Pre-Deployment Gate Summary

| Gate | Result |
|------|--------|
| Sharpe CI lower > 0 | PASS |
| Sharpe CI lower > 0.5 | **FAIL** |
| Sharpe > benchmark | PASS |
| All OOS windows Sharpe > 0 | PASS |
| Min OOS Sharpe > 0 | PASS |
| Mean OOS Sharpe > 0.5 | PASS |
| Split-sample stable (<50% decay) | PASS |
| Sensitivity robust (12/12 params) | PASS |

**Overall: FAIL — review issues below**

## 1. Statistical Significance

### Bootstrap Confidence Intervals (10k block-bootstrap, 21-day blocks)

| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |
|--------|---------------|-------------|-------------|
| Sharpe Ratio | 0.87 | 0.50 | 1.24 |
| Terminal Wealth | $15.80 | $4.46 | $52.92 |
| Max Drawdown | -23.0% | -43.9% | -17.9% |

### Ledoit-Wolf Sharpe Ratio Test (vs benchmark)

- Strategy Sharpe: 0.87
- Benchmark Sharpe: 0.57
- Difference: 0.30
- t-statistic: 1.89
- p-value: 0.0582
- Significant at 5%: **FAIL**

## 2. Walk-Forward Out-of-Sample Validation

### Split-Sample Tests

| Split | IS Sharpe | OOS Sharpe | Sharpe Decay |
|-------|----------|-----------|-------------|
| IS(2002-2010) / OOS(2010-end) | 0.84 | 0.90 | +8% |
| IS(2002-2013) / OOS(2013-end) | 0.66 | 1.02 | +53% |
| IS(2002-2016) / OOS(2016-end) | 0.73 | 1.05 | +44% |
| IS(2002-2019) / OOS(2019-end) | 0.77 | 1.09 | +43% |

### Expanding-Window Walk-Forward

| Window | Test Period | OOS Sharpe | OOS Return |
|--------|------------|-----------|-----------|
| 1 | 2007-2010 | 0.74 | 11.0% |
| 2 | 2009-2012 | 0.44 | 7.1% |
| 3 | 2011-2014 | 1.01 | 12.9% |
| 4 | 2013-2016 | 0.95 | 13.5% |
| 5 | 2015-2018 | 0.79 | 9.8% |
| 6 | 2017-2020 | 1.31 | 17.3% |
| 7 | 2019-2022 | 1.48 | 26.6% |
| 8 | 2021-2024 | 0.83 | 11.4% |
| 9 | 2023-2026 | 1.44 | 22.2% |

- Mean OOS Sharpe: 1.00
- Std OOS Sharpe: 0.33
- Min OOS Sharpe: 0.44
- % Positive: 100%

### Regime Stability

| Regime | Sharpe | Ann Return | Max DD |
|--------|--------|-----------|--------|
| Bull 2003-2007 | 1.10 | 14.3% | -11.8% |
| GFC 2008-2009 | -0.09 | -1.4% | -17.0% |
| Recovery 2009-2019 | 0.90 | 12.5% | -18.4% |
| COVID Crash | -5.29 | -233.1% | -23.0% |
| Post-COVID Bull | 1.91 | 35.5% | -10.3% |
| 2022 Bear | -1.08 | -14.3% | -15.6% |
| 2023-2024 Recovery | 1.62 | 23.7% | -11.2% |

## 3. Parameter Sensitivity Analysis

| Parameter | Values Tested | Base Sharpe | Min Sharpe | Max Sharpe | Max Change | Robust |
|-----------|-------------|------------|-----------|-----------|-----------|--------|
| baseline_ivv | 5 | 0.87 | 0.85 | 0.88 | +2% | PASS |
| baseline_qqq | 5 | 0.87 | 0.85 | 0.88 | +2% | PASS |
| faber_partial_mult | 5 | 0.87 | 0.85 | 0.88 | +1% | PASS |
| harvey_exclude_months | 5 | 0.87 | 0.86 | 0.88 | +1% | PASS |
| harvey_similar_pctl | 5 | 0.87 | 0.84 | 0.87 | +3% | PASS |
| harvey_zscore_window | 5 | 0.87 | 0.84 | 0.87 | +3% | PASS |
| max_equity | 5 | 0.87 | 0.87 | 0.87 | +0% | PASS |
| max_single | 5 | 0.87 | 0.87 | 0.87 | +0% | PASS |
| sma_periods | 6 | 0.87 | 0.80 | 0.87 | +7% | PASS |
| tier1_sub | 6 | 0.87 | 0.86 | 0.87 | +1% | PASS |
| tier2_sub | 5 | 0.87 | 0.86 | 0.87 | +0% | PASS |
| weekly_delever_threshold | 3 | 0.87 | 0.84 | 0.87 | +3% | PASS |

### baseline_ivv

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.35 | 0.88 | 11.3% | -20.3% | $12.72 |
| 0.4 | 0.87 | 11.9% | -21.6% | $14.21 |
| 0.45 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.5 | 0.86 | 12.9% | -24.2% | $17.10 |
| 0.55 | 0.85 | 12.9% | -24.8% | $17.32 |

### baseline_qqq

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.15 | 0.88 | 11.0% | -19.9% | $11.91 |
| 0.2 | 0.87 | 11.7% | -21.4% | $13.79 |
| 0.25 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.3 | 0.86 | 13.0% | -24.3% | $17.62 |
| 0.35 | 0.85 | 13.2% | -24.8% | $18.36 |

### faber_partial_mult

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.5 | 0.88 | 12.6% | -23.0% | $16.41 |
| 0.6 | 0.87 | 12.5% | -23.0% | $16.11 |
| 0.7 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.8 | 0.86 | 12.4% | -23.0% | $15.50 |
| 0.9 | 0.85 | 12.3% | -23.0% | $15.20 |

### harvey_exclude_months

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 24 | 0.87 | 12.5% | -23.0% | $16.15 |
| 30 | 0.87 | 12.6% | -23.1% | $16.22 |
| 36 | 0.87 | 12.4% | -23.0% | $15.80 |
| 42 | 0.88 | 12.5% | -22.9% | $16.20 |
| 48 | 0.86 | 12.3% | -22.9% | $15.38 |

### harvey_similar_pctl

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.1 | 0.84 | 11.9% | -25.6% | $14.09 |
| 0.12 | 0.87 | 12.6% | -25.3% | $16.21 |
| 0.15 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.18 | 0.84 | 12.2% | -23.4% | $14.68 |
| 0.2 | 0.86 | 12.5% | -23.6% | $16.06 |

### harvey_zscore_window

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 80 | 0.87 | 12.4% | -23.3% | $15.79 |
| 100 | 0.87 | 12.3% | -24.4% | $15.44 |
| 120 | 0.87 | 12.4% | -23.0% | $15.80 |
| 140 | 0.87 | 12.5% | -23.0% | $16.07 |
| 160 | 0.84 | 12.2% | -23.3% | $15.00 |

### max_equity

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.75 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.8 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.85 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.9 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.95 | 0.87 | 12.4% | -23.0% | $15.80 |

### max_single

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.5 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.55 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.6 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.65 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.7 | 0.87 | 12.4% | -23.0% | $15.80 |

### sma_periods

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| [6, 10, 12] | 0.87 | 12.4% | -23.0% | $15.80 |
| [3, 6, 10] | 0.82 | 11.4% | -25.2% | $12.50 |
| [6, 12, 18] | 0.86 | 12.3% | -23.0% | $15.39 |
| [10, 12, 15] | 0.80 | 11.9% | -23.1% | $13.63 |
| [6, 10] | 0.84 | 12.6% | -23.8% | $16.02 |
| [6, 10, 12, 18] | 0.84 | 11.9% | -23.0% | $14.04 |

### tier1_sub

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.15 | 0.87 | 12.3% | -23.0% | $15.34 |
| 0.2 | 0.87 | 12.3% | -23.0% | $15.49 |
| 0.25 | 0.87 | 12.4% | -23.0% | $15.65 |
| 0.3 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.35 | 0.86 | 12.5% | -23.0% | $15.95 |
| 0.4 | 0.86 | 12.5% | -23.0% | $16.11 |

### tier2_sub

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0.45 | 0.86 | 11.7% | -21.4% | $13.59 |
| 0.55 | 0.87 | 12.1% | -22.2% | $14.66 |
| 0.65 | 0.87 | 12.4% | -23.0% | $15.80 |
| 0.75 | 0.87 | 12.8% | -23.7% | $17.02 |
| 0.85 | 0.87 | 13.2% | -24.5% | $18.31 |

### weekly_delever_threshold

| Value | Sharpe | Ann Return | Max DD | Terminal |
|-------|--------|-----------|--------|---------|
| 0 | 0.84 | 12.4% | -23.7% | $15.38 |
| 2 | 0.86 | 12.3% | -23.0% | $15.18 |
| 3 | 0.87 | 12.4% | -23.0% | $15.80 |

## 4. Parameter Interaction Tests

Base Sharpe: 0.87

### tier1_sub x tier2_sub

| tier1_sub | tier2_sub | Sharpe | Ann Return | Max DD |
|------|------|--------|-----------|--------|
| 0.15 | 0.45 | 0.87 | 11.5% | -21.4% |
| 0.15 | 0.85 | 0.87 | 13.0% | -24.5% |
| 0.4 | 0.45 | 0.86 | 11.8% | -21.4% |
| 0.4 | 0.85 | 0.86 | 13.3% | -24.5% |

### harvey_similar_pctl x harvey_exclude_months

| harvey_similar_pctl | harvey_exclude_months | Sharpe | Ann Return | Max DD |
|------|------|--------|-----------|--------|
| 0.1 | 24 | 0.85 | 12.0% | -24.9% |
| 0.1 | 48 | 0.86 | 12.2% | -25.2% |
| 0.2 | 24 | 0.86 | 12.4% | -23.8% |
| 0.2 | 48 | 0.86 | 12.5% | -23.0% |

### baseline_ivv x baseline_qqq

| baseline_ivv | baseline_qqq | Sharpe | Ann Return | Max DD |
|------|------|--------|-----------|--------|
| 0.35 | 0.15 | 0.91 | 9.9% | -17.2% |
| 0.35 | 0.35 | 0.86 | 12.8% | -23.4% |
| 0.55 | 0.15 | 0.87 | 11.9% | -22.1% |
| 0.55 | 0.35 | 0.84 | 13.1% | -25.0% |

---
*Report generated by validation/run_all.py*