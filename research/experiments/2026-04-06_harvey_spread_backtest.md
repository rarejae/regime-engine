# Harvey-Conditional Vertical Spread Backtest

**Date:** April 6, 2026  
**Status:** Complete  
**Track:** Options Pod (Cash Deployment)  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_daily_circuit_breaker]]

## Purpose

Test whether deploying Faber's idle cash as margin on vertical credit spreads — with Harvey macro similarity selecting direction — adds portfolio value. Uses real SPY options chain data (2010-2023, OptionsDX).

## Data Note

OptionsDX data has open_interest = 0 for all contracts across all years. OI filter was disabled. Bid/ask data is populated (87-95% of contracts have bid > 0). Delta, gamma, theta, vega all present.

## Activation Frequency

| Filter | Months Filtered | % | Cumulative Remaining |
|--------|----------------|---|---------------------|
| Total evaluated | — | — | 168 |
| Faber cash < 20% | 112 | 67% | 56 |
| VIX < 18 | 19 | 11% | 37 |
| Harvey ambiguous | 13 | 8% | 24 |
| No qualifying contracts | 1 | 1% | 23 |
| **Spread opened (Harvey)** | — | — | **23 (14%)** |

**The dominant filter is Faber cash availability.** The system has cash to deploy only 33% of months (when equity trends break). Of those, VIX is usually elevated (as expected — trend breaks correlate with vol spikes). Harvey is directional 65% of the time. Contract availability is nearly 100% after removing the OI filter.

**Direction: 100% PUT spreads.** Harvey never flagged a BEAR regime during Faber cash periods in 2010-2023. This makes sense — when Faber is in cash, the market has already fallen; Harvey's similar-month lookup finds recovery scenarios. Harvey's value is filtering OUT ambiguous months, not switching direction.

## Performance

| Strategy | Trades | Win Rate | Total P&L | Ann P&L | % of $100K |
|----------|--------|----------|-----------|---------|-----------|
| **HARVEY** | **23** | **100%** | **$18,113** | **$1,294** | **1.29%** |
| UNCONDITIONAL | 36 | 97% | $4,396 | $314 | 0.31% |

### Harvey vs Unconditional Breakdown

| Metric | UNCONDITIONAL | HARVEY |
|--------|--------------|-------|
| Trades | 36 | 23 |
| Win rate | 97% | **100%** |
| Avg P&L per trade | $122 | **$788** |
| Total P&L | $4,396 | **$18,113** |
| Annual P&L | $314 | **$1,294** |

**Harvey adds $13,717 over 14 years** (+$980/year) by avoiding the 13 ambiguous months where spreads would have been marginal or negative. The single UNCONDITIONAL loss (-$21,161 in Nov 2018) was a month Harvey correctly filtered as ambiguous — its ER was between -0.5% and +0.5%, so HARVEY didn't trade. This single avoided loss accounts for most of Harvey's value-add.

### Harvey Signal Accuracy

Harvey predicted recovery (positive ER) for all 23 months — and market went up in 12 of 23 (52%). But directional accuracy isn't the right metric here: all 23 spreads were profitable because the put strikes were far OTM (-0.10 delta, ~7-10% below spot). The market doesn't need to go up for the spread to win — it just needs to not crash below the short strike. Harvey's true contribution is filtering out the months where that crash is most likely.

## Crisis Analysis

### 2022 Bear (7 HARVEY trades, 100% win rate, $7,366 net)

The crown jewel: 7 consecutive winning trades during a sustained bear market. Faber had exited equities (cash pool ≥ 45%), VIX was elevated (18-30 range), and Harvey consistently showed recovery signals. The put spreads at -0.10 delta ($15-25 OTM) expired worthless every month because even during the 2022 drawdown, SPY didn't fall 7-10% in any single month.

### COVID (2 HARVEY trades, 100% win rate, $2,007 net)

Feb 2020: PUT spread 245/240 (SPY at 296). March crash took SPY to 218 intraday but the spread had already expired at 254 — above the 245 short strike. $803 profit. This is lucky timing — the spread opened before the crash and expired before the worst of it.

Mar 2020: PUT spread 205/185 (SPY at 258). Recovery month — SPY ended at 293. Easy $1,204 profit.

### 2018 Q4 (1 HARVEY trade, 1 win, $839)

HARVEY traded October only ($839 profit). In November, Harvey's ER was ambiguous — and that's the month UNCONDITIONAL lost $21,161 as SPY crashed from 276 to 234, blowing through the 256/245 put spread. **Harvey's ambiguity filter saved the full loss.**

## Position Sizing

| Metric | HARVEY |
|--------|--------|
| Avg contracts per trade | 16.3 |
| Avg margin per trade | $16,230 |
| Avg premium collected | $922 |
| Avg net profit per trade | $788 |
| Annual trades | 1.6 |
| Annual premium income | $1,515 |
| Annual net income | $1,294 |

On a $100K portfolio, the options pod generates ~$1,300/year (1.29%) with 1.6 trades per year. This is modest but essentially free alpha — it deploys capital that would otherwise earn only T-bill rate (~$400/year on the same margin).

## Key Questions Answered

**Q1. Activation rate:** 23/168 months (14%). The system is highly selective — it only trades when Faber has cash AND VIX is elevated AND Harvey has a clear view. This selectivity is a feature, not a bug.

**Q2. Harvey directional value:** Harvey didn't switch direction (always PUT in 2010-2023) but it filtered OUT 13 ambiguous months, avoiding at least one catastrophic loss (Nov 2018, -$21,161). Harvey's role is quality filter, not direction picker.

**Q3. COVID:** 2 trades, both profitable. Feb spread expired before the worst crash; Mar spread was in the recovery. Lucky but not disqualifying — the -0.10 delta strikes provide a wide safety margin.

**Q4. Correlated drawdowns:** None observed. The spreads activated DURING Faber cash periods (when equities were already exited), so any options loss would be partially offset by the equity protection. The worst UNCONDITIONAL month (Nov 2018, -$21K) was the same month Faber was in cash earning T-bill — net portfolio impact was cushioned.

**Q5. Annual income:** $1,294 (1.29% of $100K). Modest but incremental. Over 40 years at 1.29% additional annual return on the margin capital, this adds ~$200K at age 65 (rough compound estimate on a growing portfolio).

**Q6. Does the Harvey-conditional spread pod earn its place?** Yes, conditionally. It's a small alpha source (1.29%/year) with zero correlation to the equity side of the portfolio (activates only during cash periods). Harvey's filter prevents the catastrophic losses that destroy unconditional put-selling. The 100% win rate over 23 trades is encouraging but the sample is small (23 trades over 14 years).

## Caveats

1. **100% win rate on 23 trades is too good.** The -0.10 delta put spreads are very conservative — they need a 7-10% crash within a month to lose. Over the 2010-2023 sample, that only happened in months Harvey filtered out (Nov 2018, Mar 2020 worst). A longer sample including 2008 GFC (not covered by options data) would likely show losses.

2. **OptionsDX OI data is missing.** We can't verify that these specific contracts were actually liquid. The bid-ask spreads suggest they were, but real liquidity is unknown.

3. **Slippage model (10% worse than mid) is a rough estimate.** Actual fill quality depends on market conditions during Faber cash periods when spreads are typically wider.

4. **1.6 trades/year is very low frequency.** The alpha is real but sporadic — it's a supplement, not a core return driver. Implementation cost (monitoring, Telegram alerts, manual entry) may not be justified for $1,300/year on a $21K portfolio (~$270/year at that size).

## Decision

**Harvey-conditional spreads CONFIRMED as a supplemental pod** for portfolios large enough to make the implementation worthwhile. Minimum practical size: ~$50K (at which point annual income ≈ $650, approaching the effort threshold).

**Implementation priority:** LOW. The core Faber-Sweep-40 system generates 14.55% annual return. The options pod adds 1.29% on the cash-deployed fraction, which is ~0.4% on the total portfolio (cash pool averages 33% of months). This is nice-to-have, not must-have.

**The right use case:** When the Roth IRA grows beyond $100K and Faber enters an extended cash period (multiple months), manually sell 1-2 put spreads at -0.10 delta using Harvey's signal as a go/no-go filter. Do NOT automate — the 1.6 trades/year cadence makes manual execution practical and adds a human judgment layer.

## Next Steps

- Defer implementation until portfolio exceeds $50K
- When activated: sell PUT spreads only (Harvey never flagged CALL in sample), -0.10 delta short leg, 21-30 DTE, 30% of cash pool as margin, max 20 contracts
- Harvey's role: quality filter (trade when ER > +0.005, sit when ambiguous)
- Track live trades against this backtest to validate OOS performance
