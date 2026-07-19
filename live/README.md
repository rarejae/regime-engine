# Live V19d execution

Paper twin, watcher, tax-drag, slippage stress, and Robinhood-ready broker stubs.

**Spec:** [`research/context/V19D_LIVE_EXECUTION_SPEC.md`](../research/context/V19D_LIVE_EXECUTION_SPEC.md)

## Quick start (confidence ladder)

```bash
# 1. Signal parity vs packaged research
.venv/bin/python -m live.parity

# 2. CB slippage stress report
.venv/bin/python -m live.slippage_stress

# 3. Daily watcher (dry-run CB fills → live/runtime/)
.venv/bin/python -m live.watcher
.venv/bin/python -m live.watcher --live-quotes   # refresh last prints via yfinance

# 4. Dashboard tax-drag toggle
.venv/bin/streamlit run viz/app.py
```

Default: **`DRY_RUN=true`**. Never sets live orders until Robinhood MCP is connected **and** `LIVE_TRADING=1`.

## Tiny live (Stage 3) — after you trust paper

1. In Robinhood: enable **Agentic Trading**, fund Agentic account with **$1–2.5K** only.
2. In Cursor: add MCP `https://agent.robinhood.com/mcp/trading` → OAuth.
3. Kill-switch drill: disconnect MCP + flatten in app.
4. Set stage in `live/runtime/state.json` → `"stage": 3`, keep buys manual.
5. Run watcher before 15:40 ET on trading days; CB sells pre-authorized in policy.

## Layout

| Path | Role |
|------|------|
| `signals.py` | Faber scores / CB / modes (matches research) |
| `state.py` | HOLD → CB_PENDING → FLAT → REENTRY (+ wash clock) |
| `ledger.py` | signal + fill JSONL |
| `broker.py` | DryRun + RobinhoodMCP placeholder |
| `watcher.py` | daily CLI |
| `parity.py` | research package parity |
| `slippage_stress.py` | historical CB delay cost |
| `tax.py` | taxable haircut model (used by Streamlit) |
| `runtime/` | local state/ledger (gitignored) |
| `reports/` | stress markdown/csv |

## Env

| Var | Meaning |
|-----|---------|
| `DRY_RUN` | default true — simulated fills |
| `LIVE_TRADING` | must be `1` for real broker path (still requires MCP) |
| `TELEGRAM_*` | optional later |
