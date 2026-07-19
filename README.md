# regime-engine

Quantitative TAA (tactical asset allocation) research repo. Roughly 50 logged
experiments (April 2026) that evolved a two-engine macro system into the locked
**V19d** production strategy: two Faber trend-gated 2x equity pods plus a gold
sleeve, with per-asset circuit breakers that exit to cash.

**Start here:**

- [research/context/V19D_PRODUCTION_SPEC.md](research/context/V19D_PRODUCTION_SPEC.md) — the locked production spec
- [research/experiments/V9_TO_V19D_RESEARCH_ARC.md](research/experiments/V9_TO_V19D_RESEARCH_ARC.md) — full research narrative
- [research/context/TAA_PROJECT_STATUS.md](research/context/TAA_PROJECT_STATUS.md) — chronological research timeline
- [CLAUDE.md](CLAUDE.md) — experiment protocol (signal alignment rules, vault workflow)

V19d validated performance (2002–2026): 17.27% CAGR, 0.866 Sharpe, -25.1% MaxDD.

## Repo map

### Active

| Path | Purpose |
|------|---------|
| `research/` | Obsidian vault — experiment results, decisions, context docs. Open as a vault to browse the knowledge graph. |
| `experiments/` | One folder per experiment, each with its own `backtest.py`. Mirrors `research/experiments/` notes. |
| `taa/` | TAA engine modules (Faber signals, allocation, leverage). Production implementation track. |
| `validation/` | Robustness suite (bootstrap CIs, sensitivity analysis, look-ahead audits). |
| `data/` | Fetchers (`fetcher.py`, `sources/`) and processed series. Bulk parquet/7z files are local-only (gitignored); regenerate via `data/fetcher.py` and `data/ingest_optionsdx.py`. |

### Legacy (superseded, kept for reference)

| Path | What it was | Why retired |
|------|-------------|-------------|
| `regime/`, `engine/` | Harvey-Mulliner similarity engine + Kritzman relevance engine | All macro engines destroy Sharpe vs Faber-only. See `research/context/KRITZMAN_RESEARCH_FINDINGS.md`. |
| `hmm/` | Gaussian HMM regime detection | State labeling unstable across refits; removing it had negligible impact. See `RESEARCH_HISTORY.md`. |
| `taa_v2/` | Signal-driven allocation without baseline weights | Underperformed — baseline weights are a feature. |
| `main.py`, `run_*.py`, `dashboard/` | Entry points and dashboard for the old regime engine | Tied to the retired macro engines. |
| `RESEARCH_HISTORY.md` | Findings from the pre-vault era (Faber-Harvey system) | Superseded by the vault; still the record of HMM/ensemble/carry rejections. |

The successor production repo scaffold lives at `~/Projects/faber-harvey-system`
(see git history for `migrate_manifest.txt`, which mapped the migration).

## Deferred research tracks (next up)

Ranked by leverage per dollar of data cost:

1. **Vol-managed overlay (Moreira & Muir 2017)** — scale the 2x exposure by
   inverse realized variance. Zero new data; attacks V19d's known weakness
   (first month of a crash at 180% effective equity).
2. **Managed futures pod** — DBMF (2019+) / KMLM (2020+) NAVs via yfinance,
   extended with the free SG Trend Index and the AQR TSMOM series already in
   `data/raw/aqr_tsmom_monthly.csv`. See `experiments/managed_futures_proxy/`.
3. **Merger arb pod** — MERFX daily NAV (free, 1989+) as the pod return stream;
   no HFRI subscription or deal-level data needed.
4. **Kritzman turbulence as a cross-pod risk layer** — rejected for Faber+VRP
   (too correlated), untested over genuinely uncorrelated pods. See
   `research/context/MULTI_POD_ARCHITECTURE.md`.

Dead ends are catalogued in the research arc doc — do not re-explore without
new evidence.

## Experiment visualizer

Streamlit dashboard for comparing strategies across experiments:

```bash
.venv/bin/python viz/export_v19d_marketstack.py   # once, or after new runs
.venv/bin/streamlit run viz/app.py
```

Drop future experiment packages in `viz/packages/<id>/` — see `viz/README.md`.

## Live execution (taxable / Robinhood)

See [`live/README.md`](live/README.md) and [`research/context/V19D_LIVE_EXECUTION_SPEC.md`](research/context/V19D_LIVE_EXECUTION_SPEC.md).

```bash
.venv/bin/python -m live.parity
.venv/bin/python -m live.slippage_stress
.venv/bin/python -m live.watcher          # dry-run by default
.venv/bin/streamlit run viz/app.py        # tax-drag toggle in sidebar
```
