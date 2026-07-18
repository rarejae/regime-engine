---
date: 2026-04-13
experiment: V19d-QQQ 60/30/10 QQQ Tilt Test
status: NOT PREFERRED — +0.57pp CAGR but -3.6pp MaxDD and -0.010 Sharpe
script: experiments/v19d_qqq_tilt/backtest.py
---

# V19d-QQQ: 60/30/10 QQQ Tilt — Results

## Verdict

**Not preferred.** V19d-60 gains +0.57pp CAGR and +$7.05 terminal, but at -0.010 Sharpe and -3.6pp MaxDD. The MaxDD worsening from -25.1% to -28.7% exceeds the 3pp tolerance, driven by GFC (-20.8% vs -16.4%) and COVID (-28.7% vs -25.1%).

The 60/30/10 tilt increases concentration risk during crises when QQQ falls harder than IVV. V19d's 45/45/10 is the correct balanced split.

---

## Core Metrics

| Strategy          |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 |  DCA$700 |
|-------------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|---------:|
| V19d-60 (60/30/10)| 17.84% | 22.04% |  0.856 |   0.999 | -28.7% |   0.66 | $61.65 |  $5.29M  |
| **V19d (45/45/10)**| 17.27% | 20.94% | **0.866** | **1.010** | **-25.1%** | **0.72** | $54.60 | $4.64M |

Delta: +0.57pp CAGR, -0.010 Sharpe, -3.6pp MaxDD, +$7.05 terminal.

## Crisis Drawdowns

| Crisis | V19d-60 | V19d | Delta |
|--------|--------:|-----:|------:|
| GFC    | -20.8%  | -16.4% | -4.4pp |
| COVID  | -28.7%  | -25.1% | -3.6pp |
| 2022   | -17.9%  | -17.6% | -0.3pp |

GFC and COVID are where QQQ concentration hurts most. The tilt amplifies tech-sector drawdowns during broad crises.

## Decision

Keep V19d at 45/45/10. The QQQ tilt is a valid CAGR-maximizing choice ($650K more DCA) but violates the MaxDD tolerance and trades Sharpe for return — the same tradeoff V9 makes at a larger scale. V19d's value proposition is specifically its superior Sharpe and controlled MaxDD.

V19d-60 sits between V19d and V9 on the CAGR curve. Investors wanting more CAGR than V19d should consider V9 directly ($7.37M DCA, -37.9% DD) rather than the intermediate tilt.
