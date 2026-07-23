# regime-engine

Research and backtesting for a **tactical asset allocation** (TAA) system built around Faber trend filters, leveraged equity ETFs, and cash as the defensive hedge.

After ~50 logged experiments, the locked candidate strategy is **V19d**: two independent Faber-gated 2× equity pods (QLD / SSO) plus a gold sleeve, with per-asset circuit breakers that exit to cash.

| Metric (2002–2026) | V19d |
|--------------------|-----:|
| CAGR | 17.3% |
| Sharpe | 0.87 |
| Max drawdown | −25.1%* |

\*Extended history from 2000 shows deeper drawdowns (~−40%). See the research notes.

## Live dashboard

**[Open the Streamlit visualizer →](https://rarejae-regime-engine.streamlit.app)**

Compare V19d against buy-and-hold benchmarks, inspect drawdowns, crisis windows, and allocation state.

Run locally:

```bash
.venv/bin/streamlit run viz/app.py
```

## What's in this repo

| Path | Purpose |
|------|---------|
| [`research/`](research/) | Experiment write-ups, decisions, and the production spec |
| [`experiments/`](experiments/) | Backtest scripts (one folder per experiment) |
| [`viz/`](viz/) | Streamlit dashboard + exported result packages |
| [`taa/`](taa/) | Signal / allocation engine modules |
| [`validation/`](validation/) | Robustness checks (bootstrap, sensitivity, look-ahead audits) |
| [`data/`](data/) | Data fetchers and sources |

### Start reading

1. [`research/context/V19D_PRODUCTION_SPEC.md`](research/context/V19D_PRODUCTION_SPEC.md) — locked strategy rules
2. [`research/experiments/V9_TO_V19D_RESEARCH_ARC.md`](research/experiments/V9_TO_V19D_RESEARCH_ARC.md) — how the design evolved
3. [`research/context/TAA_PROJECT_STATUS.md`](research/context/TAA_PROJECT_STATUS.md) — full research timeline

## Core idea

- **Trend filter** decides when each risk asset is eligible (price vs long SMAs)
- **Cash** absorbs freed capital when signals are off — not “defensive” substitutes
- **Leverage** (QLD / SSO) only when the trend is confirmed
- **Circuit breakers** exit a sleeve to cash if price falls below all SMAs mid-month

Macro return forecasts, portfolio-level stop overlays, and most complexity add-ons were tested and rejected. Details live under `research/decisions/` and the experiment notes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys only if regenerating market data
```

Bulk market data is gitignored; regenerate via `data/fetcher.py` when needed. The Streamlit app ships with a committed result package under `viz/packages/`.
