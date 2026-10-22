# fpso

An agentic supply chain for offshore FPSO operations. AI agents link the dependency chain that no single system holds today: a failing pump, its work order, the parts it needs, the purchase orders and suppliers behind them, and the boat that carries them offshore.

This repository is at **Phase 0**: proving that agents can build that cross-system picture from messy data. They do it by discovering links between records instead of relying on a hand-designed model. Everything in Phase 0 is mock: the operator **Northwind Offshore** and the asset **FPSO Kestrel** are fictional, and so is all the data.

## What is here

| Path | What it does |
| --- | --- |
| `src/fpso/synthetic/` | Builds FPSO Kestrel's world with a hidden ground truth, then writes it as messy ERP, P6, vessel and supplier-email extracts |
| `src/fpso/world_model/` | Postgres schema for the link store: entity pointers, links with confidence, confirmed patterns, the action log, and a chain walk |
| `src/fpso/models/` | Model client agents call by role (`adjudicate`, `extract`, `induce`), with a stub backend, an Anthropic API backend and a Claude-on-Bedrock backend |
| `src/fpso/governance/` | Action log: every model call is recorded with tokens and estimated cost |
| `src/fpso/connectors/` | Read-only connectors for the five sources (ERP maintenance, ERP materials, P6, vessel, supplier mail), each also runnable as a read-only MCP server |
| `src/fpso/resolution/` | Candidate matching (cheap, high-recall proposals with the signals that fired), Claude adjudication of each target's candidates, and per-link-type calibration with an auto-accept threshold |
| `src/fpso/onboarding.py` | Onboarding agent: drafts a plain-language description of a new source, its keys and the link types it can join |
| `src/fpso/eval/` | Scoring harness (the only code that reads `truth/`), the adjudication run, and the two simulated reviewers |
| `docs/model-setup.md` | Steps to get an Anthropic API key and point the code at it |
| `docs/aws-setup.md` | Steps to set up an AWS account for Bedrock, for a later move |

## Quick start

```bash
uv venv && uv pip install -e ".[dev,db,anthropic]"
fpso-synth --seed 7 --profile A --out data/kestrel-A   # development data
fpso-synth --seed 7 --profile B --out data/kestrel-B   # held-back data for the final run
fpso-synth --seed 7 --profile C --out data/kestrel-C   # profile A plus the fields Phase 0 found missing
fpso-replay --run data/kestrel-A --as-of 2027-02-01 --out data/kestrel-A@2027-02-01  # the extracts as of one day
fpso-eval --run data/kestrel-A                         # candidate recall + rule baseline report
fpso-adjudicate --run data/kestrel-A --sample 60        # model-in-the-loop precision/recall (capped at $5)
fpso-round --run data/kestrel-A --round 0              # measured round on the fixed gold set
fpso-onboard erp-mm --extracts data/kestrel-A/extracts  # onboarding draft for one source
fpso-readiness --run data/kestrel-A                     # weekly readiness reports, observe only
fpso-procure --run data/kestrel-A --no-model             # procurement recommendations, replayed weekly
fpso-mcp erp-pm --extracts data/kestrel-A/extracts     # serve one source over MCP (stdio)
fpso-model-check                                       # one tiny call per role (stub unless FPSO_MODEL_BACKEND=anthropic)
pytest -q
```

`FPSO_MODEL_<ROLE>` swaps the model for one role, for example `FPSO_MODEL_ADJUDICATE=claude-haiku-5-5` to try a cheaper model for matching.

Each run writes `extracts/`, which is what connectors and agents may read, and `truth/`, which is never shown to agents. The `truth/links.csv` file holds the correct cross-system links for scoring, and `truth/run.json` records the seed, the profile settings and the counts.

At full size, one run has about 300 equipment items, 190 materials, 1,500 work orders, 1,900 PO lines, 70 sailings, 2,100 manifest lines, 1,100 supplier emails and 7,200 true links. Use `--scale` for smaller runs.

### Link types scored in Phase 0

1. ERP equipment ↔ P6 activity
2. ERP work order ↔ P6 activity
3. Work-order component (reservation) ↔ PO line
4. PO line ↔ vessel manifest line
5. PO ↔ supplier email

### Corruption profiles

`A` is the development profile. It covers tag separator and case drift, missing work-order references, abbreviated manifest text, and PO numbers that appear only in email subjects. `B` uses different rules and is held back for the final measurement. It drops train suffixes, merges manifest lines, zero-pads PO numbers, puts PO numbers only in quoted replies, and writes dates day-first. `C` is profile A plus the fields the Phase 0 gate found missing: the reservation a PO line was bought for, and the PO line keys on each manifest line (each present 90% of the time). Comparing C with A shows what asking a real operator for those fields is worth. Profiles are versioned in `src/fpso/synthetic/corrupt.py`. Add a new version instead of editing an existing one.

## Adjudication run

`fpso-adjudicate` samples target records per link type (for example 60 manifest lines) and sends each one, with all its candidates and their signals, to Claude in one call. Half the targets are labelled by the simulated reviewers and used to fit an isotonic calibration and the auto-accept threshold for 95% precision. The other half are scored against `truth/`. The report gives precision of auto-accepted links, recall (true links the model did not reject), auto-resolved share, calibration error, the rule baseline on the same targets, and cost. With the stub backend every answer is "unsure", so the numbers only mean something on a real backend. `--max-cost` stops new calls once the estimated spend reaches the cap.

## Measured rounds

`fpso-round` is the harness for the plan's rounds. The first run fixes a gold set in `reports/gold.json`: 80 targets per link type, half easy and half hard, split 25% dev and 75% held-out. Every later round reuses it. On dev targets the simulated reviewers label each candidate pair; every high-confidence pair and a random 20% of the rest get a second reviewer, and pairs where they disagree, or where one of them skipped, are left out of calibration. Held-out targets are scored against `truth/`. Each round writes `reports/rounds/round-<n>.{json,md}` with the model, prompt hash and code commit, plus reviewer agreement and accuracy, and `summary.md` compares rounds. Profile B refuses to run without `--final`.

`fpso-readiness` is the Readiness agent at autonomy level 0. For each report date (by default the turnaround 6 and 5 weeks out) it takes every work order starting 4 to 26 weeks later, walks component, PO line, manifest lines and supplier emails, and gives each component at most one gap: no supply, PO late, supplier slip, or not shipped in time. Components whose gap would rest on a link it could not settle are listed as unverified instead. Links come from decisive rule matches, then the model (reusing the latest round's answers where the prompt is unchanged); the small model reads supplier emails for delays. Reports go to `reports/readiness/<date>.{json,md}` and are scored against the same facts built from the clean world: agreement is the share of work orders whose gap list matches exactly (plan target 80%).

## Procurement agent

`fpso-procure` is the Lead-time & Procurement agent at autonomy level 1: it recommends, and never orders or changes anything. For each day it replays (weekly by default, over the run's window), it reads the run as it stood that day and works back from each component's need date: the latest order date is the need minus 4 days to reach a sailing minus the material's lead time. A component with no PO line and no stock gets `order_now` once that date has passed and `order_soon` within 14 days of it. A PO line whose delivery date, or a later date from a supplier email, lands within 4 days of the need gets `expedite`. Links and email reading come from the Readiness agent's chain walk. Each day is scored against the same rules on the clean world, and `summary.json` gives precision and recall per action plus how much earlier than a planner's usual notice (2 weeks before need) each late part was flagged.

The Schedule agent runs on the same days. For each part flagged `order_now` or `expedite`, it works out how many days late the part lands, delays the work order's P6 activity by whatever the gap between need date and start does not absorb, and pushes that delay through finish-to-start successors. It reports which activities move and whether, and by how much, the turnaround end moves. The work order to activity link comes from the same chain walk; a part whose work order has no settled activity is counted as unscheduled.

## Replay clock

`fpso-replay` copies a run as it stood on one day, for agents that run daily. It keeps PO lines ordered by that day, supplier emails sent by then, and manifest lines for sailings that depart within 3 days or earlier. Masters, work orders, components, the schedule, sailings and stock stay whole, and `truth/links.csv` keeps only links whose records are visible.

## Tests

`pytest` runs everything that needs no services. The schema tests also run when `FPSO_TEST_DATABASE_URL` points at an empty Postgres database; CI provides one.
