# TAA Experiment Visualizer

Streamlit dashboard for comparing strategy backtests. Future experiments are
plug-and-play: export a package, drop it in `packages/`, reopen.

**Live:** [rarejae-regime-engine.streamlit.app](https://rarejae-regime-engine.streamlit.app)

## Quick start

```bash
# Optional: refresh packages (needs network + .env keys for Marketstack export)
.venv/bin/python viz/export_v19d_marketstack.py
.venv/bin/python viz/export_sfe_series.py

# Launch locally
.venv/bin/streamlit run viz/app.py
```

## Packages included

| Package | Contents |
|---------|----------|
| `v19d_marketstack_verification` | Locked V19d vs buy-and-hold baselines |
| `sfe_principle_series` | SFE 1/3, SFE 45/45/10, SFEv3 (+CB) vs QQQ/SPY/60-40 |

## What's included

- Strategy comparison across whichever package you select in the sidebar
- Metrics from daily returns (CAGR, Vol, Sharpe, Sortino, MaxDD, Calmar, Terminal $1)
- Custom starting principal + monthly contribution (free-form; wealth path + terminal)
- Equity curve, drawdown, annual returns, crisis table
- Allocation state and circuit-breaker event log (when the package provides them)
- Optional after-tax sensitivity toggle
- Date-range filter (default: full package window)

## Adding a future experiment

1. Produce daily and monthly return series for each strategy/benchmark.
2. Write a folder under `viz/packages/<experiment_id>/`:

```
meta.json                 # see schema.py
daily_returns.parquet     # DatetimeIndex, columns = strategy ids
monthly_returns.parquet   # month-start index, same columns
monthly_state.parquet     # optional
cb_events.csv             # optional
```

3. Restart Streamlit — the package appears in the sidebar. No app code changes.

See `schema.py` for the full contract and `export_v19d_marketstack.py` as a template exporter.
