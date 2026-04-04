# Kritzman Relevance-Weighted Allocation Engine

Experimental alternative to the Harvey-Mulliner similarity engine for the TAA system.

## Paper Reference

- Kritzman, Kulasekaran & Turkington (2023), "Portfolio Construction When Regimes Are Ambiguous", MIT Sloan Working Paper
- Czasonis, Kritzman & Turkington (2022), "Relevance", Journal of Investment Management

## How It Works

**Harvey-Mulliner**: Find the 15% most similar months by Euclidean distance in z-score space, equal-weight their forward returns.

**Kritzman Relevance**: 
1. Mahalanobis distance (accounts for indicator covariance)
2. Relevance = Similarity + Informativeness (unusual observations carry more information)
3. Top 20% most relevant months, weighted by relevance (not equal-weighted)
4. Produces regime-conditioned expected returns AND covariance matrix

## Files

| File | Purpose |
|------|---------|
| `config.py` | All parameters |
| `relevance_engine.py` | Core relevance computation (Mahalanobis, informativeness, weights) |
| `conditioned_estimates.py` | Regime-conditioned returns and covariance matrix |
| `allocation.py` | Three allocation schemes: inverse-vol, mean-variance, risk parity |
| `backtest.py` | Full monthly backtest harness |
| `comparison.py` | Visualization: equity curves, drawdowns, scatter plots |

## Running

```bash
# Full backtest + report
python experiments/kritzman_relevance/backtest.py

# Generate comparison charts (after backtest)
python experiments/kritzman_relevance/comparison.py
```

## Key Differences from Harvey

| Aspect | Harvey-Mulliner | Kritzman Relevance |
|--------|----------------|-------------------|
| Distance metric | Euclidean | Mahalanobis |
| Selection | Bottom 15% by distance | Top 20% by relevance |
| Weighting | Equal | Relevance-proportional |
| Unusual observations | Penalized (far from current) | Rewarded (informative) |
| Output | Expected returns only | Returns + full covariance matrix |
| Allocation | Inverse-vol only | Inv-vol, mean-variance, risk parity |
