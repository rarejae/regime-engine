# V19d Live Execution Spec

**Status:** Active — agentic loop, human approval on buys  
**Account:** Taxable Robinhood Agentic sleeve  
**Related:** [[V19D_PRODUCTION_SPEC]] | [[2026-07-18_marketstack_verification]]

---

## Purpose

The agent runs V19d. You only see approval requests. Circuit-breaker sells flatten to cash without asking. Slippage is not tracked and is not a gate.

Stateful MCP is fine: the agent runs in a Cursor session with Robinhood connected. Strategy state lives in `live/runtime/`, not in the MCP session.

---

## Approval matrix

| Action | Agent | You |
|--------|-------|-----|
| CB sell → cash | Executes | — |
| Monthly buy / re-lever / re-entry | Drafts ticket | Approve or reject |
| Mode-change rebalance | Drafts ticket | Approve or reject |
| Wash-sale blocked | Informs | — |
| Options | Never | — |

Rejecting a buy keeps that sleeve flat until the **mode changes** (no daily nag).

Kill switch: disconnect Robinhood MCP + flatten Agentic in the app.

---

## Fill policy

### Circuit breaker — no ask
1. 3/3 SMA breach on QQQ / IVV / IAU (sleeve held).
2. Sell to **cash** (not unlevered ETF). Prefer into the close; else next session market.
3. Wash-sale clock (31d) starts on a loss-taking CB sell. CB is never delayed for tax.

### Buys / rebalance — you approve
1. Agent sizes from `capital` in `live/runtime/state.json` (45/45/10, partial = 70% of sleeve).
2. Ticket sits in `live/runtime/approvals.json` until you approve or reject.
3. Approve → agent places the buy. Reject → sleeve stays cash until mode changes.

---

## Taxable rules

`tax_mode = TAXABLE_STANDARD`

1. CB exits are never delayed for tax.
2. After a loss-taking CB sell of T, no buy of T (or substantially identical) for 31 calendar days.
3. QLD/SSO stay in the Agentic taxable sleeve.

---

## Signals and data

- Same SMA periods `[126, 200, 252]` as `experiments/v19d_final`.
- **History / SMAs:** yfinance. `--live-quotes` refreshes **IVV, QQQ, IAU, QLD, SSO**. Cache still uses SPY/GLD as long-history proxies.
- **Quotes at fill:** Robinhood when MCP is connected; yfinance otherwise.
- MCP is not the SMA source.

---

## State machine

```
HOLD → CB_PENDING → FLAT → REENTRY_ELIGIBLE → HOLD
         ↑                        │
         └──── (new breach) ──────┘
```

- `REENTRY_ELIGIBLE` = ticket waiting for you. `HOLD` only after approve (or CB path).
- Persisted in `live/runtime/state.json`. Repeated runs upsert the same ticket; they do not double-sell.

---

## Robinhood

- MCP: `https://agent.robinhood.com/mcp/trading` (OAuth; not in-repo).
- Trades only the funded Agentic account. Start with test capital ($1k).
- `DRY_RUN=true` until MCP is connected **and** `LIVE_TRADING=1`.

---

## Daily loop

```bash
.venv/bin/python -m live.watcher --live-quotes   # auto CB; queue buys
.venv/bin/python -m live.approvals list
.venv/bin/python -m live.approvals approve pod1-buy pod2-buy
```

In Cursor: agent runs the watcher, executes CB via MCP, then **stops and asks** with ticket ids.

---

## Non-goals

- No slippage budgets, shadow fills, or paper-vs-live slip audit
- No discretionary “wait for a better open”
- No options
- No unofficial Robinhood APIs
- No unattended cron orders until MCP can run outside a Cursor session
