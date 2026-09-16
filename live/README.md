# Live V19d

Two execution paths. **Fidelity HSA (manual)** is the current production target —
tax-free wrapper, no tax drag, no wash-sale. The Robinhood agentic path is kept
for reference.

**Spec:** [`research/context/V19D_LIVE_EXECUTION_SPEC.md`](../research/context/V19D_LIVE_EXECUTION_SPEC.md)

---

## Fidelity HSA — manual placement + alerts (`live/fidelity.py`)

Fidelity has no agentic/official trading API, so the repo never places orders. It
computes the V19d target from a data source, tells you exactly what to trade, and
tracks the portfolio itself so it knows current weights — a Fidelity feed is only
needed to reconcile, not to run.

```bash
# fund it (record your contribution)
.venv/bin/python -m live.fidelity add-cash 5000

# daily — CB sells surface any day; a full rebalance only on the first trading day
.venv/bin/python -m live.fidelity plan --live-quotes

# after you place the trades in the Fidelity app/web
.venv/bin/python -m live.fidelity confirm

# anytime
.venv/bin/python -m live.fidelity status
.venv/bin/python -m live.fidelity positions ~/Downloads/Portfolio_Positions.csv
```

What each run does:
- **Circuit breaker (any day):** if a held sleeve's asset closes below all 3 SMAs,
  the plan shows an **urgent** SELL-to-cash — place it that session.
- **Month start:** full rebalance to 45/45/10 target modes (buys, re-levers, drift
  trims > 5%). Off month-start with no CB → *hold*.
- **Alerts:** every plan prints, is appended to `live/runtime/alerts/<date>.txt`, and
  — if `ALERT_WEBHOOK` is set — POSTs `{"text": …}` (Slack/Discord/Telegram shape).
- **State:** `confirm` advances `live/runtime/fidelity_state.json` (per-sleeve shares
  + cash). `positions` reconciles it from a Fidelity CSV export.

No wash-sale clock and no tax drag — it's a tax-free account, so V19d runs at its
full pre-tax profile.

---

## Robinhood agentic loop (reference) — `live/watcher.py`

The agent runs V19d; you see only **approval requests**. CB sells auto-execute.
Requires the Robinhood Trading MCP connected in Cursor and `LIVE_TRADING=1`
(default `DRY_RUN=true`).

```bash
.venv/bin/python -m live.watcher --live-quotes
.venv/bin/python -m live.approvals list
.venv/bin/python -m live.approvals approve pod1-buy pod2-buy
```

| Action | Who |
|--------|-----|
| CB sell → cash | Agent, no ask |
| Monthly buy / re-lever / re-entry | You approve |
| Drift / mode-change rebalance | You approve |
| Options | Never |

---

## Data

- **Signals (SMAs):** yfinance history via the shared cache
  (`experiments/v19d_marketstack_verification/yf_cache.parquet`). `--live-quotes`
  refreshes IVV/QQQ/IAU/QLD/SSO last prints.
- MCP is never the SMA source.

## Layout

| Path | Role |
|------|------|
| `fidelity.py` | **Fidelity HSA manual planner** — plan / confirm / status / positions |
| `alerts.py` | console + dated file + optional webhook delivery |
| `signals.py` | Faber scores / CB / modes (shared) |
| `sizing.py` | 45/45/10 sleeve notionals (shared) |
| `ledger.py` | activity log (shared) |
| `watcher.py` | Robinhood agentic loop: auto CB, queue approvals |
| `approvals.py` | Robinhood approval queue |
| `state.py` | Robinhood state machine + wash-sale clocks |
| `broker.py` | DryRun + Robinhood MCP placeholder |

## Env

| Var | Meaning |
|-----|---------|
| `ALERT_WEBHOOK` | optional — Slack/Discord/Telegram incoming webhook for Fidelity alerts |
| `DRY_RUN` | Robinhood path only — default true |
| `LIVE_TRADING` | Robinhood path only — must be `1` for the real broker path |
