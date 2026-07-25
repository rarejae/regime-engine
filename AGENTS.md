# AGENTS.md

## Cursor Cloud specific instructions

This repo is a Python research/backtesting project (`regime-engine`) plus two dashboards. The startup update script already creates the Python venv (`.venv`), installs `requirements.txt`, and installs the React dashboard deps. Below are the non-obvious things to know when running/testing.

### Services

| Service | Location | Run command | Notes |
|---------|----------|-------------|-------|
| Streamlit visualizer (primary product) | `viz/app.py` | `.venv/bin/streamlit run viz/app.py` | Reads committed result packages under `viz/packages/`. No API keys or network needed. |
| React "Faber" dashboard (secondary) | `dashboard/` | `npm --prefix dashboard run dev` | Standalone Vite app; all backtest data is inlined in `dashboard/FaberDashboard.jsx`. No API keys needed. |
| Tests | `tests/` | `.venv/bin/python -m pytest` | 147 tests, all offline (network sources are mocked). Emits many deprecation warnings — harmless. |
| Macro Regime CLI | `main.py`, `run_backtest.py`, `run_full_audit.py`, `run_regime_audit.py` | `.venv/bin/python main.py` | Requires `FRED_API_KEY` (see `.env.example`) AND live network (FRED / yfinance). Not runnable in a sandboxed/offline env without those. |

### Gotchas

- No linting is configured (no ruff/flake8/pyproject config). "Lint" is not applicable.
- `python`/`pip` are not on PATH; use `python3` and the venv binaries (`.venv/bin/...`).
- `dashboard/node_modules` is committed to git but contains macOS (`darwin-arm64`) native binaries. On Linux this makes `vite` fail with `ERR_MODULE_NOT_FOUND`. The update script fixes this by removing `dashboard/node_modules` and reinstalling for the current platform. Do NOT commit the resulting `node_modules`/`package-lock.json` churn — it is platform-specific noise.
- Bulk market data (`data/processed/`, `data/raw/...`) is gitignored and regenerated via `data/fetcher.py` (needs API keys). The dashboards do not depend on it — they use the committed `viz/packages/` parquet files and inlined JSX data.
- Streamlit is configured headless via `.streamlit/config.toml`; run with `--server.port 8501` and open `http://localhost:8501`.
