# Multi-Agent Fraud Detection for Reply Mirror 2026

This repository contains an agent-based fraud detection system built for the Reply AI Challenge 2026 "Reply Mirror" scenario. The project combines deterministic fraud analytics with LangGraph orchestration, OpenRouter-hosted LLM reasoning, and Langfuse challenge tracing.

The goal is to identify suspicious transactions in evolving, adversarial financial data while keeping false positives under control and preserving the agentic nature required by the competition.

## Table of Contents

- [Challenge Context](#challenge-context)
- [Project Goals](#project-goals)
- [System Architecture](#system-architecture)
- [Dataset Overview](#dataset-overview)
- [Repository Structure](#repository-structure)
- [Core Components](#core-components)
- [Technical Stack](#technical-stack)
- [Tracing and Submission Requirements](#tracing-and-submission-requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Evaluation and Testing](#evaluation-and-testing)
- [Output Format](#output-format)
- [Known Limitations](#known-limitations)
- [Design Decisions](#design-decisions)
- [References](#references)

## Challenge Context

Reply Mirror is a fraud detection challenge set in the year 2087. Teams are asked to build agent-based systems that detect suspicious financial behavior from heterogeneous sources:

- transactions
- user profiles
- geolocation traces
- SMS threads
- email threads

The challenge explicitly rewards:

- fraud detection accuracy
- low false positive rate
- operational speed and cost efficiency
- robustness to temporal and structural drift
- agent-based design rather than fully deterministic pipelines

Competition constraints captured in this codebase:

- output must be an ASCII text file containing suspicious `transaction_id` values
- one transaction ID per line
- reporting no IDs is invalid
- reporting all IDs is invalid
- recall below 15% is invalid
- top solutions may be reevaluated on unseen datasets

The official challenge brief is mirrored in `problemoverview/AIAgentChallenge-ProblemStatement16April.md`.

## Project Goals

This repository is designed around four engineering goals:

1. Use deterministic code where exact computation is clearly better than LLM reasoning.
2. Use specialist agents only where interpretation, ambiguity, or evidence fusion matter.
3. Keep the system auditable with structured state, structured agent outputs, and reproducible traces.
4. Balance fraud recall with cost and runtime constraints required by the challenge.

## System Architecture

The system is implemented as a LangGraph workflow over a shared `FraudState` object defined in `state.py`.

High-level topology:

```text
START
  |
  v
Node 1: Deterministic Featurizer
  |
  +--> Node 2: Transaction Reasoning Agent
  |
  +--> Node 3: Communications Reasoning Agent
              |
              v
Node 4: Supervisor Agent
  |
  v
END -> submission output file
```

There is also a short-circuit route intended to bypass specialist agents when deterministic evidence is already decisive.

### Shared State

`state.py` defines the contract that all nodes read and update. It includes:

- transaction identity and raw transaction context
- matched user profile
- deterministic risk features
- transaction-agent outputs
- communications-agent outputs
- final supervisor verdict and explanation

Important fields include:

- `velocity_score`
- `amount_zscore`
- `balance_integrity_flag`
- `iban_risk_tier`
- `geo_travel_anomaly`
- `demographic_deviation_pct`
- `drift_psi`
- `extracted_entities`
- `cross_source_flags`
- `combined_risk_score`
- `transaction_fraud_signal`
- `comms_fraud_signal`
- `verdict`

## Dataset Overview

The root dataset directory is `train-validation/`.

Supported levels in the current loader:

- `brave-new-world`
- `deus-ex`
- `the-truman-show`

Each level has a `train` and `validation` zip. The loader in `data_loader.py` resolves paths such as:

- `Brave+New+World+-+train.zip`
- `Deus+Ex+-+validation.zip`
- `The+Truman+Show+-+validation.zip`

Each zip contains five sources:

### `transactions.csv`

Structured transaction ledger with fields such as:

- `transaction_id`
- `sender_id`
- `recipient_id`
- `transaction_type`
- `amount`
- `location`
- `payment_method`
- `sender_iban`
- `recipient_iban`
- `balance_after`
- `description`
- `timestamp`

### `users.json`

User profile records including:

- name
- birth year
- salary
- job
- IBAN
- residence coordinates
- natural-language profile description

### `locations.json`

GPS pings keyed by `biotag`, with:

- timestamp
- latitude
- longitude
- city

### `sms.json`

SMS thread payloads stored under a single `sms` string field.

### `mails.json`

Email or HTML mail thread payloads stored under a single `mail` field.

## Repository Structure

Main project files:

- `pipeline.py` - generic CLI runner for a given level and split
- `submit.py` - validation/submission runner
- `evaluate.py` - local evaluation harness
- `calibrate.py` - threshold calibration script
- `data_loader.py` - dataset loading and normalization
- `featurizer.py` - deterministic feature engineering node
- `graph.py` - LangGraph assembly
- `session.py` - session ID generation and Langfuse config
- `state.py` - shared state schema

Agents:

- `agents/transaction_agent.py`
- `agents/comms_agent.py`
- `agents/supervisor_agent.py`

Prompts:

- `prompts/transaction_agent_prompt.py`
- `prompts/comms_agent_prompt.py`
- `prompts/supervisor_prompt.py`

Deterministic tools:

- `tools/transaction_tools.py`
- `tools/geospatial_tools.py`
- `tools/comms_tools.py`

Tests:

- `tests/test_featurizer.py`
- `tests/test_agents.py`
- `tests/test_graph.py`
- `tests/test_state.py`
- `tests/test_evaluate.py`

Reference and support material:

- `problemoverview/`
- `reply-mirror-multi-agent-fraud-design.md`
- `COMPARISON_REPORT.md`
- `Langfuse/how-to-track-your-submission/`

## Core Components

### 1. Deterministic Featurizer

Implemented in `featurizer.py`.

This node computes structured fraud indicators before any LLM call is made. It exists to keep exact arithmetic, data matching, and statistical logic outside the LLM loop.

Current featurizer responsibilities:

- match users by `sender_iban` against `users.json`
- compute sender transaction velocity
- compute amount anomaly z-score
- detect suspicious balance drain patterns
- classify IBAN country risk
- detect impossible travel from GPS pings
- score demographic deviation from salary, job, amount, and hour
- measure sender drift using PSI-like scoring over transaction amounts
- extract structured entities from SMS and mail threads
- detect cross-source mismatches between communications and transaction details
- compute a `combined_risk_score`

The communications preprocessing intentionally uses regex-based extraction rather than spaCy, because this repo notes installation friction for the original spaCy plan in `tools/comms_tools.py`.

### 2. Transaction Reasoning Agent

Implemented in `agents/transaction_agent.py`.

This agent receives deterministic transaction features and synthesizes a structured fraud assessment:

- `transaction_fraud_signal`
- `pattern_label`
- `reasoning`

It uses:

- `ChatOpenAI` through OpenRouter
- a structured system prompt
- JSON-only parsing with retry-on-invalid-output
- the `FAST_MODEL` environment variable

### 3. Communications Reasoning Agent

Implemented in `agents/comms_agent.py`.

This agent analyzes free-text communication evidence and extracted entities to identify:

- urgency language
- impersonation clues
- IBAN mismatches
- suspicious URLs
- social engineering patterns

It returns:

- `comms_fraud_signal`
- `flagged_phrases`
- `cross_reference_mismatches`
- `reasoning`

It uses the `STRONG_MODEL` environment variable because communications reasoning is more language-heavy than the structured transaction agent.

### 4. Supervisor Agent

Implemented in `agents/supervisor_agent.py`.

This agent fuses the deterministic features and both specialist outputs into a final verdict:

- `FRAUD`
- `REVIEW`
- `CLEAN`

It also returns:

- `confidence`
- `primary_evidence`
- `explanation`

The output writer in `graph.py` includes both `FRAUD` and `REVIEW` IDs in the submission file, while suppressing `CLEAN`.

### 5. LangGraph Workflow

Implemented in `graph.py`.

This file defines:

- featurizer node
- conditional routing after featurization
- transaction specialist node
- communications specialist node
- supervisor node
- short-circuit supervisor node
- output writer

It is also where specialist parallelism and short-circuiting are orchestrated.

## Technical Stack

The project currently depends on:

- Python 3.12+ in the working environment
- LangGraph
- LangChain
- LangChain OpenAI integration
- Langfuse SDK v3
- pandas
- numpy
- scipy
- scikit-learn
- haversine
- python-dotenv
- python-ulid
- pytest

The exact declared dependencies are in `requirements.txt`.

## Tracing and Submission Requirements

The challenge expects Langfuse tracing and model access through OpenRouter.

This repository supports that through `session.py`, which:

- creates a session ID in the format `{TEAM_NAME}-{ULID}`
- configures a Langfuse callback handler for LangChain
- injects `langfuse_session_id` into LangChain metadata
- flushes Langfuse traces after a run

The helper material in `Langfuse/how-to-track-your-submission/` is a reference example. The actual project integration used by the pipeline is `session.py`.

## Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Create a `.env` file

This repository expects a dotenv file at the project root named `.env`.

Recommended variables:

```env
OPENROUTER_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
LANGFUSE_MEDIA_UPLOAD_ENABLED=false
TEAM_NAME=your-team-name
DATA_DIR=train-validation
FAST_MODEL=openai/gpt-4o-mini
STRONG_MODEL=openai/gpt-4o
SHORTCIRCUIT_THRESHOLD=0.90
AGENT_MODEL=openai/gpt-4o-mini
```

## Configuration

Important runtime knobs:

- `TEAM_NAME`
  Used in the Langfuse session ID.
- `DATA_DIR`
  Directory containing the train/validation zip files.
- `FAST_MODEL`
  Used by the transaction reasoning agent.
- `STRONG_MODEL`
  Used by the communications and supervisor agents.
- `SHORTCIRCUIT_THRESHOLD`
  Intended cutoff for deterministic-only routing when the featurizer is confident enough.

## How to Run

### Run the general pipeline

Train split:

```powershell
python pipeline.py --level brave-new-world --split train
```

Validation split:

```powershell
python pipeline.py --level the-truman-show --split validation
```

Smoke test with a small subset:

```powershell
python pipeline.py --level the-truman-show --split validation --max-transactions 5
```

### Generate a submission-style output file

For validation output:

```powershell
python submit.py --level the-truman-show
```

Smoke test:

```powershell
python submit.py --level the-truman-show --max-transactions 5
```

Expected output path:

```text
outputs\the-truman-show_validation_output.txt
```

### Run calibration

If you want to sweep thresholds over training data:

```powershell
python calibrate.py
```

Check the script itself for level-specific behavior and any output artifacts it writes.

## Evaluation and Testing

### Local evaluation

The evaluation harness compares a generated output file against a reference labels file.

Example:

```powershell
python evaluate.py --level brave-new-world --split train
```

If labels are not supplied, the script reports output statistics only.

### Tests

Run the project test suite with:

```powershell
python -m pytest -q tests
```

This is the recommended test target. Running plain `pytest` at repository root may also collect unrelated tests inside reference repositories under `repos/`.

## Output Format

Submission files are ASCII text files with:

- one suspicious `transaction_id` per line
- no header
- no footer

The writer in `graph.py` currently outputs both:

- `FRAUD`
- `REVIEW`

and suppresses:

- `CLEAN`

## Known Limitations

This section is intentionally candid so that readers and judges understand the current state of the repository.

### 1. End-to-end pipeline runtime bug in `graph.py`

Targeted tests pass, but a real pipeline smoke run currently fails in the graph dispatch path with a LangGraph node return-value error.

Observed behavior:

- the deterministic loading and featurization run correctly
- the pipeline enters the graph
- the dispatch path returns `Send(...)` objects in a way LangGraph rejects at runtime

Practical effect:

- `submit.py` and `pipeline.py` are not yet reliably producing valid end-to-end fraud predictions on real runs
- fallback handling may still emit `REVIEW` output entries after errors

This is the main blocker to producing a proper challenge submission from the current codebase.

### 2. Communications preprocessing is regex-based

The design material originally considered spaCy-based entity extraction, but the implemented code uses regex and curated phrase matching in `tools/comms_tools.py`.

This keeps the system lightweight and dependency-friendly, but it is less expressive than a full NLP parser.

### 3. Reference repos are included in-tree

The `repos/` directory contains comparative/reference projects and is not part of the main runtime path. Be careful not to confuse them with the production code for this submission.

## Design Decisions

This repository intentionally uses a hybrid approach rather than a pure multi-agent swarm.

### Why deterministic features first

Fraud signals such as:

- rolling velocity
- z-scores
- IBAN country parsing
- GPS speed calculations
- balance arithmetic

are exact, cheap, and easier to validate in Python than in LLM prompts.

### Why only two specialist agents

The architecture aims for a practical middle ground:

- enough specialization to satisfy the challenge's agent-based requirement
- not so many agents that cost, latency, and orchestration overhead dominate

### Why a supervisor agent

Naive averaging or voting is usually too brittle for asymmetric fraud costs. The supervisor lets the system reason over disagreement, corroboration, and borderline cases with a structured verdict.

### Why model tiering exists

The repository separates:

- a faster model for structured transaction reasoning
- a stronger model for language-heavy communications and final arbitration

This is a cost-performance optimization inspired by the comparison work documented in `COMPARISON_REPORT.md`.

## References

Project-specific references in this repository:

- `problemoverview/AIAgentChallenge-ProblemStatement16April.md`
- `reply-mirror-multi-agent-fraud-design.md`
- `COMPARISON_REPORT.md`
- `Langfuse/how-to-track-your-submission/README.txt`

## Suggested Next Steps

The most useful next engineering steps for this repository are:

1. Fix the LangGraph specialist dispatch path in `graph.py` so real end-to-end runs complete.
2. Benchmark multiple `STRONG_MODEL` options on a small training subset.
3. Calibrate fraud and review thresholds using `calibrate.py`.
4. Add a smoke test that exercises the real graph execution path rather than only mocked specialist nodes.
