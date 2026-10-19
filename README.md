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
fpso-eval --run data/kestrel-A                         # candidate recall + rule baseline report
fpso-adjudicate --run data/kestrel-A --sample 60        # model-in-the-loop precision/recall (capped at $5)
fpso-round --run data/kestrel-A --round 0              # measured round on the fixed gold set
fpso-onboard erp-mm --extracts data/kestrel-A/extracts  # onboarding draft for one source
fpso-mcp erp-pm --extracts data/kestrel-A/extracts     # serve one source over MCP (stdio)
fpso-model-check                                       # one tiny call per role (stub unless FPSO_MODEL_BACKEND=anthropic)
pytest -q
```

Each run writes `extracts/`, which is what connectors and agents may read, and `truth/`, which is never shown to agents. The `truth/links.csv` file holds the correct cross-system links for scoring, and `truth/run.json` records the seed, the profile settings and the counts.

At full size, one run has about 300 equipment items, 190 materials, 1,500 work orders, 1,900 PO lines, 70 sailings, 2,100 manifest lines, 1,100 supplier emails and 7,200 true links. Use `--scale` for smaller runs.

### Link types scored in Phase 0

1. ERP equipment ↔ P6 activity
2. ERP work order ↔ P6 activity
3. Work-order component (reservation) ↔ PO line
4. PO line ↔ vessel manifest line
5. PO ↔ supplier email

### Corruption profiles

`A` is the development profile. It covers tag separator and case drift, missing work-order references, abbreviated manifest text, and PO numbers that appear only in email subjects. `B` uses different rules and is held back for the final measurement. It drops train suffixes, merges manifest lines, zero-pads PO numbers, puts PO numbers only in quoted replies, and writes dates day-first. Profiles are versioned in `src/fpso/synthetic/corrupt.py`. Add a new version instead of editing an existing one.

## Adjudication run

`fpso-adjudicate` samples target records per link type (for example 60 manifest lines) and sends each one, with all its candidates and their signals, to Claude in one call. Half the targets are labelled by the simulated reviewers and used to fit an isotonic calibration and the auto-accept threshold for 95% precision. The other half are scored against `truth/`. The report gives precision of auto-accepted links, recall (true links the model did not reject), auto-resolved share, calibration error, the rule baseline on the same targets, and cost. With the stub backend every answer is "unsure", so the numbers only mean something on a real backend. `--max-cost` stops new calls once the estimated spend reaches the cap.

## Measured rounds

`fpso-round` is the harness for the plan's rounds. The first run fixes a gold set in `reports/gold.json`: 80 targets per link type, half easy and half hard, split 25% dev and 75% held-out. Every later round reuses it. On dev targets the simulated reviewers label each candidate pair; every high-confidence pair and a random 20% of the rest get a second reviewer, and pairs where they disagree are left out of calibration. Held-out targets are scored against `truth/`. Each round writes `reports/rounds/round-<n>.{json,md}` with the model, prompt hash and code commit, plus reviewer agreement and accuracy, and `summary.md` compares rounds. Profile B refuses to run without `--final`.

## Tests

`pytest` runs everything that needs no services. The schema tests also run when `FPSO_TEST_DATABASE_URL` points at an empty Postgres database; CI provides one.
