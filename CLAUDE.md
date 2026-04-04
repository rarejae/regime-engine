# Regime Engine

Macroeconomic regime engine for tactical asset allocation. Two-step Faber-Harvey signal system with graduated leverage overlay.

## Quick Start

```bash
python main.py              # Full CLI run, rich-formatted TradeCard
python main.py --json       # Machine-readable JSON
python run_dashboard.py     # Live terminal dashboard
python run_backtest.py      # Dual-window backtest
```

## Project Structure

```
regime/           Core regime detection (Faber filter, Harvey similarity, HMM, Kritzman)
  run_*.py        Signal development backtest runners → output to experiments/signal_development/
engine/           ConditionVector pipeline: classify → score → construct trade
backtest/         Backtest framework (runner, metrics, trade simulator)
data/             Data ingestion, caching, source adapters (FRED, Yahoo, BLS, CBOE, etc.)
  macro/          Parquet files for macro indicators and asset returns
hmm/              Hidden Markov Model package
taa/              Tactical asset allocation v1
taa_v2/           Tactical asset allocation v2
validation/       Walk-forward, sensitivity, statistical significance
config/           YAML config files (indicators, regimes, settings)
experiments/      All experiments organized by theme (see below)
tests/            Main test suite (pytest)
```

## Experiments

All experiment scripts and outputs live under `experiments/`. Each has an `output/` subfolder for reports, charts, and parquet artifacts.

| Directory | What it tests |
|-----------|--------------|
| `signal_development/` | Core signal iteration: Faber, HMM, Kritzman, ensemble, hierarchical v1-v4, blend, strategy layer, tactical overlay, IWM, Roth portfolios. Output-only archive from `regime/run_*.py` scripts. |
| `kritzman_relevance/` | Alternative allocation engine using Kritzman relevance-weighted estimates + MVO |
| `proxy_validation/` | Pre-ETF proxy quality: IEF proxy testing, replacement search, comprehensive audit |
| `asset_universe/` | Universe configs: 4-asset, 5-asset, treasury bond integration (TNX, IEF, pipeline) |
| `leverage_tiers/` | Leverage tier analysis: conservative vs aggressive, middle ground |
| `portfolio_diagnostics/` | Crisis weight behavior, portfolio beta, component value-add, weekly vs monthly rebalancing |

## Running Experiments

All experiment scripts are run from the project root:

```bash
python experiments/leverage_tiers/leverage_test.py
python experiments/kritzman_relevance/backtest.py
```

## Environment

- Python 3.x with `.venv/` virtual environment
- Required secret: `FRED_API_KEY` in `.env`
- Optional: `SCHWAB_APP_KEY`, `SCHWAB_APP_SECRET`
- Tests: `pytest tests/`

## Key Architecture Decisions

- **ConditionVector** (`engine/condition_vector.py`) is the central data structure for all trade decisions
- Regime labels (GOLDILOCKS, etc.) are display-only, not used for allocation
- Faber SMA filter (6/10/12-month) gates asset eligibility; Harvey similarity directs freed capital
- Graduated leverage (1.0x/1.3x/1.65x) with weekly 3-of-3 SMA circuit breaker
- All parameters derived from published methodology, not optimized on backtest data
- Point-in-time discipline: every signal uses `.shift(1)` to prevent look-ahead bias

## Conventions

- Experiment scripts use `sys.path.insert(0, ...)` to find project root for `regime.*` imports
- Output paths use `Path(__file__).resolve().parent / "output"` for experiment-local output
- Data paths (`data/macro/`) are project-root-relative (scripts run from project root)
