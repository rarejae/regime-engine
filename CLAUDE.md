# CLAUDE.md

## Before Starting Any Task

Read `research/context/TAA_PROJECT_STATUS.md` for current project state, active architecture, what has been tested, and current conclusions. Do not re-test approaches already rejected unless explicitly asked.

## Signal Alignment (CRITICAL)

Every backtest must enforce these with code assertions:

- Month T signals use only data available through month T (or T-1 for macro indicators)
- Allocation decisions apply to month T+1 returns
- Expanding windows only — no look-ahead in z-scores, covariance matrices, or rolling statistics
- No concurrent return contamination

## Running Experiments

- Always include standard benchmarks for comparison (IVV B&H, 60/40, and the current best-performing system documented in the status file)
- Report: annualized return, vol, Sharpe, Sortino, max DD, terminal $1
- Include crisis analysis: GFC (2008-2009), COVID (2020), 2022 Bear
- Default to 1x leverage unless explicitly testing leverage

## After Every Experiment

1. Write results to `research/experiments/YYYY-MM-DD_experiment_name.md` using the template in `research/templates/experiment_template.md`
2. Update `research/context/TAA_PROJECT_STATUS.md` — add to the research timeline, update conclusions if changed
3. If rejecting an approach, create a decision record in `research/decisions/` using the template in `research/templates/decision_template.md`
4. Use `[[wiki links]]` for all cross-references between documents
5. Print results to terminal as well for immediate review

## Code Structure

- `taa/` — Production module
- `experiments/` — Research experiments (separate from production)
- `research/` — Obsidian vault for findings, decisions, and project state
