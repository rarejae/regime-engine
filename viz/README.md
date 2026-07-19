# TAA Experiment Visualizer

Streamlit dashboard for comparing strategy backtests. Designed so future
experiments are plug-and-play: export a package, drop it in `packages/`, reopen.

## Quick start

```bash
# 1. Export the latest Marketstack verification package (needs network + .env keys)
.venv/bin/python viz/export_v19d_marketstack.py

# 2. Launch
.venv/bin/streamlit run viz/app.py
```

## What's included

- Strategy comparison: V19d, IVV B&H, QQQ B&H, 50/50 IVV/QQQ, 60/40
- Metrics from daily returns (CAGR, Vol, Sharpe, Sortino, MaxDD, Calmar, Terminal $1)
- DCA / contributions: presets + interactive start capital and monthly contribution
- Equity curve, drawdown, annual returns, crisis table
- V19d allocation state (effective equity + Faber scores) and CB event log
- Date-range filter (default: full package window, e.g. 2000→present)

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
