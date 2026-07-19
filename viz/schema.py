"""Experiment package schema (v1) — plug-and-play for the Streamlit visualizer.

Layout
------
viz/packages/<experiment_id>/
  meta.json                 required
  daily_returns.parquet     required — DatetimeIndex, one column per strategy id
  monthly_returns.parquet   required — month-start DatetimeIndex, same columns
  monthly_state.parquet     optional — primary strategy state (scores, modes)
  cb_events.csv             optional — circuit breaker log

meta.json
---------
{
  "schema_version": "1.0",
  "experiment_id": "v19d_marketstack_verification",
  "title": "V19d Marketstack Verification",
  "description": "...",
  "generated": "2026-07-18T...",
  "date_start": "2000-01-01",
  "date_end": "2026-07-17",
  "primary_strategy": "v19d",
  "strategies": [
    {"id": "v19d", "name": "V19d", "kind": "strategy", "color": "#2563eb"},
    {"id": "ivv_bh", "name": "IVV B&H", "kind": "benchmark", "color": "#64748b"},
    ...
  ],
  "data_notes": "Marketstack 2016+ spliced with yfinance...",
  "source_experiment": "experiments/v19d_marketstack_verification"
}

To add a future experiment
--------------------------
1. Produce daily + monthly return series for each strategy/benchmark.
2. Write them into a new folder under viz/packages/<id>/ matching this schema.
3. The dashboard auto-discovers packages — no app code changes required.
"""

SCHEMA_VERSION = "1.0"

REQUIRED_FILES = ("meta.json", "daily_returns.parquet", "monthly_returns.parquet")
OPTIONAL_FILES = ("monthly_state.parquet", "cb_events.csv")
