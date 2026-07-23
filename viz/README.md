# TAA Experiment Visualizer

Streamlit dashboard for comparing strategy backtests. Future experiments are
plug-and-play: export a package, drop it in `packages/`, reopen.

**Live:** [rarejae-regime-engine.streamlit.app](https://rarejae-regime-engine.streamlit.app)

## Quick start

```bash
# Optional: refresh the Marketstack verification package (needs network + .env keys)
.venv/bin/python viz/export_v19d_marketstack.py

# Launch locally
.venv/bin/streamlit run viz/app.py
```

## What's included

- Strategy comparison: V19d vs IVV / QQQ / 50/50 / 60/40
- Metrics from daily returns (CAGR, Vol, Sharpe, Sortino, MaxDD, Calmar, Terminal $1)
- Equity curve, drawdown, annual returns, crisis table
- V19d allocation state and circuit-breaker event log
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
