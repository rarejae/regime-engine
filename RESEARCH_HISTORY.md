# Faber-Harvey System — Research History

## Final Architecture

Two-step hierarchical signal system with graduated leverage. Step 1: Multi-timeframe Faber SMA trend filter (6/10/12-month) gates asset eligibility and frees capital. Step 2: Harvey-Mulliner regime similarity engine directs freed capital via inverse-volatility scoring. Graduated leverage overlay (1.0x/1.3x/1.65x) engages when both signals converge, with weekly 3-of-3 SMA breach circuit breaker.

**Production performance (2002-2026):** Sharpe 0.87, annualized return 12.4%, max drawdown -23.0%, terminal $15.80 from $1.

## Rejected Approaches

### Signal Layer

**HMM regime detection** — Built 2-state, 3-state, and 4-state Gaussian HMMs with rolling window fitting across 8 macro features. State labeling was fundamentally unstable: states would invert between refits (what was "bull" becomes "bear"), requiring post-hoc correction via excess return analysis. Even with inversion detection, HMM added noise to Harvey signals. The final hierarchical system (Faber→Harvey→HMM→Kritzman) showed that removing HMM entirely had negligible impact on Sharpe. *Key learning: probabilistic regime models need enormous data to be stable; with ~20 years of monthly data, the estimation error dominates the signal.*

**Ensemble voting (4-method)** — Combined Harvey similarity, Faber trend, HMM state, and Kritzman turbulence into a composite score. Each method voted bull/bear/neutral per asset. Result: worse than Faber alone (Sharpe 0.82 vs 0.90). The problem is methodological: averaging a reliable signal (Faber) with unreliable ones (HMM, Kritzman) dilutes the reliable signal. *Key learning: signal combination only helps when all signals have independent, positive information ratios.*

**Carry and value signals** — Implemented per-asset carry (yield spread) and value (real yield z-score) for bonds, gold, and commodities. VGLT carry had a sign error (positive when bonds were expensive), and the value signal z-scores were unstable across regimes. After fixing the sign, carry/value added <0.02 Sharpe improvement while doubling code complexity. *Key learning: carry/value are powerful in multi-asset macro funds with 20+ instruments; with 5 assets, the diversification benefit is too small to justify.*

**Kritzman turbulence and absorption ratio** — Mahalanobis distance of daily cross-asset returns (turbulence) and PCA variance ratio (absorption ratio). Turbulence correctly identified GFC (z=+3.76) but was too late — by the time turbulence spikes, the drawdown is already happening. Absorption ratio was degenerate with 5 assets (99%+ explained variance at all times). Tried as a "brake" overlay on Harvey: reduced Sharpe from 0.87 to 0.81 by forcing unnecessary de-risking. *Key learning: turbulence is a coincident indicator, not a leading one. Works for intraday risk management, not monthly allocation.*

**Strategy layer with floors and ceilings** — Tilt-based allocation with per-asset floors (5%) and ceilings (40%) that widened or narrowed based on regime conviction. Created a DBC 86% concentration problem: the cap-then-normalize loop pushed weight back above ceiling, requiring iterative fixing. Even after fixing, the band system was a complicated way to express what Faber filter + Harvey direction already captured. *Key learning: explicit constraints (Faber eligibility gates) are cleaner than soft bands.*

### Baseline & Allocation

**Inverse-volatility baseline weights** — Computed weights from pre-2002 realized volatility (1/vol, normalized). Result: 69.4% cash, 14.1% DBC, 10.4% IAU, 2.9% IVV, 1.2% QQQ. The "portfolio" was a savings account. Sharpe improved to 1.05 (trivially — just the cash-loading Sharpe trick) but terminal wealth collapsed from $15.80 to $4.35 (-73%). *Key learning: inverse-vol naively applied pushes everything to the lowest-vol asset. Only works when applied within asset classes (e.g., among equities only), not across asset classes with wildly different vol profiles.*

**Remove 40% single-asset concentration cap** — Tested removing the cap in `direct_capital()` and `normalize()` that limits any single asset to 40% when it's the only eligible Harvey target. Performance was identical (Sharpe 0.87 → 0.86) but crisis drawdowns worsened: GFC max DD -7.7% → -10.6%, 2022 bear DD -15.6% → -19.1%. The cap binds in 87% of months but its protection is concentrated in exactly the months that matter. *Key learning: concentration caps are free insurance — they cost nothing in normal times and save 3pp in crises.*

**Signal-driven allocation (no baseline, taa_v2/)** — Removed fixed baseline weights entirely; let graduated Faber multipliers (0.15/0.70/1.00) and Harvey scores determine allocation from scratch. Tested three universe configs (A: IVV+QQQ+IAU, B: IVV+IAU, C: IVV+QQQ+VGLT+IAU). All underperformed the baseline system. The baseline captures equity risk premium in every month — removing it means you only earn premium when signals are strongly bullish, missing the ~7%/yr earned during "neutral" months. *Key learning: baseline weights are a feature, not a constraint. They represent a prior that equities earn a premium, which signals can tilt but shouldn't override.*

**IWM in universe** — Added Russell 2000 (IWM) as a 6th risky asset. Harvey directed capital to IWM in recovery periods, but IWM's higher vol and lower Sharpe ratio dragged overall performance. Removing IWM improved Sharpe by 0.04. *Key learning: adding assets only helps if they have independent positive expected returns; IWM is too correlated with IVV to add diversification value.*

### Leverage

**Flat 1.25x leverage** — Applied uniform 1.25x to IVV/QQQ during all conviction periods via synthetic SSO/QLD substitution. Sharpe 0.84 (worse than 1x at 0.85) because leverage amplified drawdowns in low-conviction months that happened to pass the basic filter. *Key learning: uniform leverage destroys risk-adjusted returns; graduation is essential.*

**Conservative tiers (1.25x/1.5x)** — First graduated attempt: Tier 1 at 25% SSO/QLD substitution, Tier 2 at 50%. Terminal $13.40, Sharpe 0.84. The median threshold for Tier 2 was initialized at 0, causing all early Tier 1 months to auto-qualify for Tier 2 — a bug that inflated early performance. After fixing the median seed (pre-backtest unconditional mean Harvey ER), recalibrated to 1.3x/1.65x (30%/65% substitution) which achieved Sharpe 0.87 and terminal $15.80. *Key learning: the median initialization bug was invisible in backtest results (it improved returns!) but would have caused excessive leverage in live trading during the seed period.*

**Weekly 1-of-3 SMA de-lever** — Friday close check: if either IVV or QQQ below ANY of the 3 SMAs, de-lever. Produced 41 events (1.7/year) — chronic whipsaw. Most events were false alarms during normal pullbacks, forcing unnecessary cash drag. Tightened to 2-of-3 (14 events) and 3-of-3 (14 events, fewer false alarms). 3-of-3 won: same number of events but better signal quality — only triggers during genuine trend breaks. *Key learning: the threshold between "useful circuit breaker" and "whipsaw generator" is narrow. Requiring ALL signals to agree (3/3) is much more robust than requiring any (1/3).*

## Key Learnings

1. **Bugs trace to guessed parameters.** The expanding median seed (initialized at 0), the Faber look-ahead (same-month price for same-month allocation), the Harvey forward return lookup (concurrent instead of T+1) — every major bug came from a parameter or timing assumption that "seemed right" but wasn't verified. The fix was always the same: derive from data or published methodology, never guess.

2. **Simplicity wins.** Two published, peer-reviewed signals (Faber trend filter, Harvey regime similarity) beat every complex combination we tried. Adding HMM, Kritzman, carry/value, ensemble voting — each added code and none added Sharpe. The final system has 5 core files totaling ~600 lines.

3. **Parameters before backtests.** Every parameter in the production system was set before seeing backtest results: SMA periods from Faber (2007), similarity percentile from Harvey & Mulliner, z-score window from standard macro practice. The sensitivity analysis confirmed: Sharpe varies only 0.80-0.91 across all 65+ parameter perturbations. When parameters come from methodology rather than optimization, there's nothing to overfit.

4. **Baseline weights are a feature.** The subjective 70% equity baseline encodes "equities earn a risk premium" — the single strongest prior in asset allocation. Every attempt to remove or weaken it (signal-driven, inverse-vol) reduced terminal wealth. Signals should tilt the baseline, not replace it.

5. **Point-in-time discipline catches real bugs.** Every signal shift (Faber .shift(1), Harvey z_data.shift(1), forward return .shift(-1)) prevented a look-ahead bias that inflated backtested Sharpe by 0.3-0.9. The discipline of asking "what would I know at market close on day T?" for every data access caught 5 separate timing bugs.

6. **Graduated leverage requires graduated exits.** Flat leverage with binary exit (on/off) creates whipsaw. Graduated leverage (Tier 0/1/2) with weekly monitoring and a high-threshold circuit breaker (3-of-3 SMA breach) achieves the return benefit of leverage without the volatility penalty.

## Validation Summary

Full robustness validation suite run on production parameters (2002-2026):

- **Bootstrap CI (10k block-bootstrap, 21-day blocks):** Sharpe [0.50, 1.24], terminal wealth [$4.46, $52.92]
- **Ledoit-Wolf test vs IVV:** Sharpe diff +0.30, p=0.058
- **Walk-forward (9 expanding windows, 3yr OOS):** Mean OOS Sharpe 1.00, min 0.44, 9/9 positive
- **Split-sample (4 cuts):** OOS consistently outperforms IS (no overfitting signal)
- **Parameter sensitivity (12 parameters, 65+ perturbations):** 12/12 robust, Sharpe range 0.80-0.91
- **Parameter interactions (3 pairs, corner cases):** All corners Sharpe 0.84-0.91
- **Crisis performance:** GFC -7.7% DD (vs IVV -50%), COVID -23.0% DD, 2022 -15.6% DD
- **Forward Sharpe estimate:** 0.70-0.85 (accounting for transaction costs, execution slippage, and the lower bound of the bootstrap CI)
