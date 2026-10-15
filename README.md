# fpso

An agentic supply chain for offshore FPSO operations. AI agents link the dependency chain that no single system holds today: a failing pump, its work order, the parts it needs, the purchase orders and suppliers behind them, and the boat that carries them offshore.

This repository is at **Phase 0**: proving that agents can build that cross-system picture from messy data. They do it by discovering links between records instead of relying on a hand-designed model. Everything in Phase 0 is mock: the operator **Northwind Offshore** and the asset **FPSO Kestrel** are fictional, and so is all the data.

## What is here

| Path | What it does |
| --- | --- |
| `src/fpso/synthetic/` | Builds FPSO Kestrel's world with a hidden ground truth, then writes it as messy ERP, P6, vessel and supplier-email extracts |
| `src/fpso/world_model/` | Postgres schema for the link store: entity pointers, links with confidence, confirmed patterns, the action log, and a chain walk |
| `src/fpso/models/` | Model client agents call by role (`adjudicate`, `extract`, `induce`), with a stub backend and a Claude-on-Bedrock backend |
| `src/fpso/governance/` | Action log: every model call is recorded with tokens and estimated cost |
| `docs/aws-setup.md` | Steps to set up the AWS account for Bedrock |

## Quick start

```bash
uv venv && uv pip install -e ".[dev,db,bedrock]"
fpso-synth --seed 7 --profile A --out data/kestrel-A   # development data
fpso-synth --seed 7 --profile B --out data/kestrel-B   # held-back data for the final run
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

## Tests

`pytest` runs everything that needs no services. The schema tests also run when `FPSO_TEST_DATABASE_URL` points at an empty Postgres database; CI provides one.
