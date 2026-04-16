# Multi-Agent Fraud Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-1c7c54.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.15-1c7c54.svg)](https://github.com/langchain-ai/langchain)
[![LangSmith](https://img.shields.io/badge/LangSmith-0.7.32-1c7c54.svg)](https://smith.langchain.com/)
[![Langfuse](https://img.shields.io/badge/Langfuse-3.14.6-6b21a8.svg)](https://langfuse.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-via%20OpenRouter-412991.svg)](https://openrouter.ai)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-1.41.0-f5a800.svg)](https://opentelemetry.io/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.2-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-f7931e.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.17.1-8caae6.svg)](https://scipy.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8.14-09a3d5.svg)](https://spacy.io/)
[![Haversine](https://img.shields.io/badge/Haversine-2.9.0-4b5563.svg)](https://pypi.org/project/haversine/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.13.1-e92063.svg)](https://docs.pydantic.dev/)
[![Rich](https://img.shields.io/badge/Rich-15.0.0-ff6600.svg)](https://github.com/Textualize/rich)
[![tqdm](https://img.shields.io/badge/tqdm-4.67.3-ffc107.svg)](https://tqdm.github.io/)
[![python-dotenv](https://img.shields.io/badge/python--dotenv-1.2.2-ecd53f.svg)](https://pypi.org/project/python-dotenv/)
[![pytest](https://img.shields.io/badge/pytest-9.0.3-0a9edc.svg)](https://pytest.org/)

**Reply Mirror Hackathon — Problem Statement 16 April 2026**

A production-quality multi-agent fraud detection pipeline built on LangGraph that combines deterministic feature engineering with LLM-powered reasoning to detect Business Email Compromise (BEC) and payment-redirect fraud in a fictional banking dataset.

---

## Table of Contents

1. [Background & Problem](#1-background--problem)
2. [System Architecture](#2-system-architecture)
3. [Feature Engineering & Formulations](#3-feature-engineering--formulations)
4. [Agent Design](#4-agent-design)
5. [Decision Logic & Thresholds](#5-decision-logic--thresholds)
6. [Prerequisites](#6-prerequisites)
7. [Installation](#7-installation)
8. [Configuration](#8-configuration)
9. [Data Setup](#9-data-setup)
10. [Quickstart](#10-quickstart)
11. [CLI Reference](#11-cli-reference)
12. [Available Models](#12-available-models)
13. [Evaluation & Metrics](#13-evaluation--metrics)
14. [Threshold Calibration](#14-threshold-calibration)
15. [Submission](#15-submission)
16. [Testing](#16-testing)
17. [Observability with Langfuse](#17-observability-with-langfuse)
18. [Performance Optimizations](#18-performance-optimizations)
19. [Project Structure](#19-project-structure)

---

## 1. Background & Problem

### Fraud Type: Business Email Compromise / Reply-Chain Fraud

**Business Email Compromise (BEC)** is a social-engineering attack in which adversaries infiltrate or spoof legitimate email threads to redirect payments. The FBI IC3 report (2023) classifies BEC as the costliest cybercrime category, responsible for over \$2.9 billion in losses annually.

The **Reply Mirror** scenario models a variant called *reply-chain invoice fraud*:

1. An attacker intercepts an ongoing payment thread (SMS or email).
2. They inject a fraudulent message that mimics the counterparty and substitutes an IBAN.
3. The victim authorises a transfer believing it is part of an existing conversation.

### Why Multi-Agent?

Single-model approaches fail because fraud signals are **heterogeneous and correlated across dimensions**:

| Signal Type | Source | Example |
|---|---|---|
| Behavioural | Transaction history | Sudden velocity spike |
| Statistical | Amount distribution | Z-score > 3σ |
| Geospatial | GPS + transaction location | 1 200 km/h implied travel |
| Linguistic | SMS / email text | "Urgent: account will be closed" |
| Cross-referential | Comms ↔ transaction | IBAN in message ≠ IBAN in transaction |

A supervisor agent arbitrates specialist signals with **asymmetric cost weighting**: missing a fraud is more costly than a false positive, so thresholds lean toward recall over precision.

### Literature Basis

| Technique | Reference |
|---|---|
| Population Stability Index (PSI) for drift | Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley |
| Haversine impossible-travel detection | Óskarsdóttir et al. (2019). *The value of big data for credit scoring.* KAIS |
| LLM agents for fraud reasoning | Bhatt et al. (2024). *LLM4Fraud: Large language models for financial fraud detection.* |
| Multi-agent orchestration | Hong et al. (2023). *MetaGPT: Meta programming for multi-agent collaborative framework.* arXiv:2308.00352 |
| ReAct prompting | Yao et al. (2022). *ReAct: Synergizing reasoning and acting in language models.* arXiv:2210.03629 |
| Structured Chain-of-Thought | Wei et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS |
| AML typology (FATF/Egmont) | FATF (2020). *Virtual Assets Red Flag Indicators.* |

---

## 2. System Architecture

### Graph Topology

```
START
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  NODE 1 — FEATURIZER  (deterministic, no LLM)           │
│  Computes 15 features → combined_risk_score             │
└─────────────────────────┬───────────────────────────────┘
                          │
               ┌──────────▼──────────┐
               │  CONDITIONAL EDGE   │
               │  CRS ≥ threshold?   │
               └──────┬──────────────┘
          YES ←───────┤───────────→ NO
          │           │             │
          ▼           │    ┌────────▼─────────────────────┐
┌─────────────────┐   │    │  NODE 2 — TRANSACTION AGENT  │
│  NODE 4         │   │    │  FAST_MODEL (gpt-4o-mini)    │
│  SUPERVISOR     │   │    │  Signal 0–1 + pattern label  │
│  (short-circuit)│   │    └────────────────────────────┬─┘
└─────────────────┘   │    ┌──────────────────────────┐ │
                      │    │  NODE 3 — COMMS AGENT    │ │
                      │    │  STRONG_MODEL (gpt-4o)   │ │
                      │    │  Signal 0–1 + mismatches │ │
                      │    └─────────────────────────┬┘ │
                      │                              │   │
                      │    ┌─────────────────────────▼───▼────┐
                      │    │  NODE 4 — SUPERVISOR AGENT       │
                      └───►│  STRONG_MODEL (gpt-4o)           │
                           │  Verdict: FRAUD | REVIEW | CLEAN  │
                           └──────────────────────────────────┘
                                          │
                                        END
                           Flagged IDs → ASCII output file
```

### State Layers

The pipeline threads a single `FraudState` TypedDict through every node:

```
Layer 1 — Identity (immutable input)
  transaction_id, user_id, raw_transaction, raw_user

Layer 2 — Deterministic Features (populated by Featurizer)
  velocity_score, amount_zscore, balance_integrity_flag,
  iban_risk_tier, geo_travel_anomaly, demographic_deviation_pct,
  drift_psi, extracted_entities, cross_source_flags,
  combined_risk_score

Layer 3 — Specialist Outputs (populated by Transaction + Comms agents)
  transaction_fraud_signal (0–1), transaction_pattern_label,
  transaction_reasoning, comms_fraud_signal (0–1),
  flagged_phrases, cross_reference_mismatches, comms_reasoning

Layer 4 — Final Verdict (populated by Supervisor)
  verdict ("FRAUD"|"REVIEW"|"CLEAN"), confidence (0–1),
  primary_evidence, explanation
```

---

## 3. Feature Engineering & Formulations

All features are computed deterministically in `featurizer.py` before any LLM call.

### 3.1 Velocity Score

Count of transactions by the same `sender_id` within a ±24-hour window centred on the current transaction timestamp:

```
velocity = |{tx_j : sender_j = sender_i  ∧  |t_j − t_i| ≤ 24h}|
```

Thresholds: ≥ 5 transactions → elevated; ≥ 10 → high risk.

### 3.2 Amount Z-Score

Standardised deviation of the transaction amount against the sender's historical distribution:

```
z = |amount_i − μ_sender| / (σ_sender + ε)
```

where ε = 1.0 prevents division by zero for senders with a single transaction.  
Additionally, the **salary ratio** `amount / monthly_salary` is computed to flag amounts exceeding typical income (> 1× salary = elevated; > 3× = high).

### 3.3 Balance Integrity Flag

Three drain patterns are detected:

| Pattern | Condition |
|---|---|
| Near-zero drain | `balance_after ≤ 200` AND `balance_before ≥ 1 000` |
| Large-fraction drain | Single transaction > 80% of `balance_before` |
| Consecutive violations | 2 or more of the above in sender's recent history |

The flag is `True` if any pattern is present. The 5% threshold used in earlier designs was too sensitive because intermediate deposits inflate running balances.

### 3.4 IBAN Risk Tier

The first two characters of an IBAN encode the ISO 3166-1 alpha-2 country code. Each country is classified by FATF/AML risk:

```python
HIGH_RISK   = {"NG", "UA", "BY", "KP", "IR", "SY", "YE", "LY", "MM", "AF"}
MEDIUM_RISK = {"RU", "VN", "PK", "BD", "GH", "SN", "TZ", "KE", "PH", "MX"}
# All others → LOW_RISK
```

### 3.5 Geo Travel Anomaly (Haversine)

Given the Haversine distance `d(A, B)` in km and the time gap `Δt` in hours, the implied travel speed is:

```
speed = d(last_GPS_ping, transaction_location) / Δt  [km/h]
```

The flag is `True` when `speed > 900 km/h` (supersonic threshold; commercial aviation ≈ 900 km/h). The haversine formula:

```
a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
d = 2·R·arcsin(√a),   R = 6371 km
```

### 3.6 Demographic Deviation Score

A weighted sum (0–1) of atypicality signals relative to the sender's profile:

| Signal | Weight |
|---|---|
| `amount > 3× monthly_salary` | +0.35 |
| `amount > 1× monthly_salary` | +0.15 |
| Transaction hour in [23:00–05:00] | +0.20 |
| Occupation = retired/student + large e-commerce | +0.20 |

Score is clamped to [0, 1].

### 3.7 Pattern Drift (PSI — Population Stability Index)

PSI measures distributional shift of sender transaction amounts between a *baseline* window (older 70%) and a *recent* window (newest 30%), using equal-frequency binning over 10 bins:

```
PSI = Σᵢ (Actual%ᵢ − Expected%ᵢ) · ln(Actual%ᵢ / Expected%ᵢ)
```

Interpretation:
- PSI < 0.10 → No significant change
- 0.10 ≤ PSI < 0.25 → Moderate shift (monitor)
- PSI ≥ 0.25 → **Significant drift** (flag)

### 3.8 Combined Risk Score

A normalised weighted sum of all deterministic signals, used as a prior for the LLM agents and as the short-circuit trigger:

```
CRS = clip(
  0.20·v_norm + 0.15·z_norm + 0.20·balance_flag +
  0.15·iban_risk + 0.20·geo_flag + 0.10·demog +
  0.10·psi_flag + 0.10·cross_flag,
  0, 1
)
```

where each term is normalised to [0, 1] before weighting.

### 3.9 Communications Entity Extraction

Regex patterns applied to raw SMS and email text:

| Entity | Pattern |
|---|---|
| IBAN | `[A-Z]{2}\d{2}[A-Z0-9]{4,30}` |
| Amount | `[€$£][\d,]+` or `[\d,]+\s?(?:EUR\|USD\|GBP\|CHF)` |
| URL | `https?://[^\s"<>]+` |
| Urgency phrase | hardcoded list: *urgent, immediately, asap, verify account, account will be closed, …* |

Cross-reference mismatches (comms IBAN ≠ transaction IBAN, comms amount ≠ transaction amount) are passed to both the Comms agent and Supervisor.

---

## 4. Agent Design

### 4.1 Transaction Reasoning Agent

- **Model**: `FAST_MODEL` (default: `openai/gpt-4o-mini`)
- **Framework**: LangGraph `create_react_agent` (ReAct pattern)
- **Prompt style**: 4-state Structured Chain-of-Thought (SCoT)

**Reasoning states:**

| State | Goal |
|---|---|
| 1 — Signal Inventory | List all anomaly magnitudes; anchor to `combined_risk_score` |
| 2 — Pattern Classification | Map signals to one of 6 fraud patterns |
| 3 — Corroboration Check | Count mutually corroborating signals; calibrate score |
| 4 — JSON Output | Emit structured verdict |

**Six fraud patterns recognised:**

| Label | Description |
|---|---|
| `velocity_fraud` | > 5 transactions in 24 h |
| `account_takeover` | Sudden behaviour change + impossible travel or balance drain |
| `mule_activity` | Money-laundering relay pattern |
| `identity_mismatch` | IBAN country ≠ user residence country |
| `synthetic_identity` | Implausible/constructed user profile |
| `unclear` | Signals present but no clear pattern |

**Output schema:**

```json
{
  "transaction_fraud_signal": 0.0,
  "pattern_label": "velocity_fraud",
  "reasoning": "One-sentence rationale."
}
```

### 4.2 Communications Reasoning Agent

- **Model**: `STRONG_MODEL` (default: `openai/gpt-4o`)
- **Framework**: LangGraph `create_react_agent` with 2 tool calls
- **Prompt style**: 4-state SCoT with few-shot examples

**Reasoning states:**

| State | Goal |
|---|---|
| 1 — Entity Review | Examine IBANs, amounts, URLs, urgency phrases from comms |
| 2 — Cross-Reference | Compare comms entities against transaction record |
| 3 — Linguistic Risk | Urgency, impersonation, social-engineering narrative |
| 4 — JSON Output | Emit structured verdict |

**Output schema:**

```json
{
  "comms_fraud_signal": 0.0,
  "flagged_phrases": ["Urgent: verify immediately"],
  "cross_reference_mismatches": ["IBAN in SMS differs from tx recipient IBAN"],
  "reasoning": "One-sentence rationale."
}
```

### 4.3 Supervisor Agent

- **Model**: `STRONG_MODEL`
- **Framework**: Direct LLM invoke (no tool loop)
- **Role**: Cost-aware arbitrator with hard decision rules

**Reconciliation logic:**

| Evidence pattern | Decision guidance |
|---|---|
| Both signals ≥ 0.60 + CRS ≥ 0.50 | Strong corroboration → FRAUD |
| CRS ≥ 0.85 (deterministic certainty) | Trust arithmetic/physics → FRAUD |
| One signal ≥ 0.60 OR avg ≥ 0.50 | Partial evidence → REVIEW |
| All signals < 0.35 + CRS < 0.40 + no flags | Low risk → CLEAN |
| Default (ambiguous) | Err on caution → REVIEW |

**Output schema:**

```json
{
  "verdict": "FRAUD",
  "confidence": 0.87,
  "primary_evidence": "IBAN substitution in SMS + impossible travel.",
  "explanation": "One-sentence audit trail."
}
```

---

## 5. Decision Logic & Thresholds

### Hard Decision Rules (Supervisor)

```
FRAUD  ← (tx_signal ≥ 0.60 AND comms_signal ≥ 0.60 AND CRS ≥ 0.50)
          OR (CRS ≥ 0.85)

REVIEW ← tx_signal ≥ 0.60
          OR comms_signal ≥ 0.60
          OR (tx_signal + comms_signal) / 2 ≥ 0.50
          OR (CRS ≥ 0.65 AND max(tx_signal, comms_signal) ≥ 0.40)

CLEAN  ← tx_signal < 0.35
          AND comms_signal < 0.35
          AND CRS < 0.40
          AND no cross_source_flags
          AND no comms mismatches

DEFAULT → REVIEW  (asymmetric cost: missing fraud > false positive)
```

### Short-Circuit Rule

```
IF CRS ≥ SHORTCIRCUIT_THRESHOLD (default 0.90):
    skip Transaction agent + Comms agent entirely
    run Supervisor directly on deterministic evidence
    → ~75% reduction in LLM calls for high-risk transactions
```

---

## 6. Prerequisites

- Python 3.11+
- Access to [OpenRouter](https://openrouter.ai/) (for LLM routing) OR direct Anthropic / OpenAI API keys
- Langfuse account credentials (provided by the hackathon organisers)
- The competition data ZIP files (provided separately)

---

## 7. Installation

```bash
# Clone or unzip the project
cd Multi_Agent_Fraud_Detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Core dependencies

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | 1.1.6 | Multi-agent graph orchestration |
| `langchain-openai` | 1.1.14 | LLM client (OpenRouter gateway) |
| `langfuse` | 3.14.6 | Tracing and observability |
| `pandas` | 3.0.2 | DataFrame operations |
| `numpy` | 2.4.4 | Numerical computations |
| `haversine` | 2.9.0 | Geospatial distance |
| `scikit-learn` | 1.8.0 | Metrics (precision/recall/F1) |
| `python-dotenv` | 1.2.2 | `.env` loading |
| `python-ulid` | 3.1.0 | Unique session IDs |
| `pytest` | 9.0.3 | Test suite |

---

## 8. Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env   # if example exists
# or create manually:
```

```ini
# ── LLM routing via OpenRouter ──────────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE

# Model for Transaction agent (fast + cheap)
FAST_MODEL=openai/gpt-4o-mini

# Model for Comms agent and Supervisor (strong reasoning)
STRONG_MODEL=openai/gpt-4o

# ── Langfuse observability ───────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY=pk-lf-YOUR_KEY
LANGFUSE_SECRET_KEY=sk-lf-YOUR_KEY
LANGFUSE_HOST=https://challenges.reply.com/langfuse
LANGFUSE_MEDIA_UPLOAD_ENABLED=false

# ── Competition settings ─────────────────────────────────────────────────────
TEAM_NAME=Your-Team-Name
DATA_DIR=train-validation

# ── Feature/decision tuning ──────────────────────────────────────────────────
SHORTCIRCUIT_THRESHOLD=0.90
```

### Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter gateway API key (required) |
| `FAST_MODEL` | `openai/gpt-4o-mini` | Model for Transaction agent |
| `STRONG_MODEL` | `openai/gpt-4o` | Model for Comms agent + Supervisor |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse instance URL |
| `TEAM_NAME` | `team` | Prefix for session IDs in Langfuse |
| `DATA_DIR` | `train-validation` | Directory containing data ZIP files |
| `SHORTCIRCUIT_THRESHOLD` | `0.90` | CRS threshold to skip specialist agents |

---

## 9. Data Setup

The competition provides data in a ZIP archive. Unzip it to match this structure:

```
train-validation/
├── brave-new-world/
│   ├── train/
│   │   ├── transactions.csv
│   │   ├── users.json
│   │   ├── locations.json
│   │   ├── sms.json
│   │   └── mails.json
│   └── validation/
│       └── (same files, no labels)
├── deus-ex/
│   └── ...
└── the-truman-show/
    └── ...
```

### Data schema

**`transactions.csv`**

| Column | Type | Description |
|---|---|---|
| `transaction_id` | str | Unique transaction ID |
| `sender_id` | str | Sender user ID |
| `recipient_id` | str | Recipient user ID |
| `type` | str | Transaction type (e.g. TRANSFER, PAYMENT) |
| `amount` | float | Transaction amount (EUR) |
| `location` | str | Transaction location city |
| `payment_method` | str | Method (e.g. ONLINE, POS) |
| `sender_iban` | str | Sender's IBAN |
| `recipient_iban` | str | Recipient's IBAN |
| `balance_after` | float | Sender's balance after transaction |
| `description` | str | Free-text description |
| `timestamp` | datetime | ISO 8601 timestamp |

**`users.json`** — dict keyed by `user_id`; fields: `first_name`, `last_name`, `birth_year`, `salary`, `job`, `iban`, `residence.city`, `residence.lat`, `residence.lng`

**`locations.json`** — list of GPS pings: `biotag` (user_id), `timestamp`, `lat`, `lng`, `city`

**`sms.json` / `mails.json`** — dict keyed by `transaction_id` (or user identifier); values are raw text threads

---

## 10. Quickstart

```bash
# 1. Run the pipeline on the training split (brave-new-world level)
python pipeline.py --level brave-new-world --split train

# 2. Evaluate against ground-truth labels
python evaluate.py --level brave-new-world --split train

# 3. Calibrate thresholds (requires label file)
python calibrate.py --level brave-new-world --labels-file outputs/brave-new-world_train_labels.txt

# 4. Submit on validation split
python submit.py --level brave-new-world
```

Output is written to `outputs/brave-new-world_train_output.txt` — one `transaction_id` per line for every transaction flagged as FRAUD or REVIEW.

---

## 11. CLI Reference

### `pipeline.py` — Main inference runner

```
python pipeline.py [OPTIONS]

Options:
  --level TEXT              Competition level (required)
                            Choices: brave-new-world | deus-ex | the-truman-show
  --split TEXT              Dataset split (default: train)
                            Choices: train | validation
  --data-dir TEXT           Path to data directory (default: env DATA_DIR or train-validation)
  --output-dir TEXT         Output directory (default: outputs)
  --max-transactions INT    Cap number of transactions (useful for smoke testing)

Output:
  outputs/{level}_{split}_output.txt  — flagged transaction IDs, one per line
```

**Examples:**

```bash
# Full training run
python pipeline.py --level brave-new-world --split train

# Smoke test: first 20 transactions only
python pipeline.py --level brave-new-world --split train --max-transactions 20

# Validation split with custom output dir
python pipeline.py --level deus-ex --split validation --output-dir my-outputs

# All three levels
for level in brave-new-world deus-ex the-truman-show; do
  python pipeline.py --level $level --split train
done
```

---

### `evaluate.py` — Evaluation harness

```
python evaluate.py [OPTIONS]

Options:
  --level TEXT              Competition level (required)
  --split TEXT              Dataset split (default: train)
  --output-dir TEXT         Where to read pipeline output from (default: outputs)
  --labels-file TEXT        Path to ground-truth label file
                            (default: outputs/{level}_{split}_labels.txt)
  --data-dir TEXT           Data directory (default: env DATA_DIR or train-validation)

Prints:
  Precision, Recall, F1, False-Positive Rate
  Warning if Recall < 15% (competition disqualification threshold)
```

**Examples:**

```bash
# Evaluate with default label path
python evaluate.py --level brave-new-world --split train

# Evaluate with explicit label file
python evaluate.py --level brave-new-world --split train \
  --labels-file /path/to/ground_truth.txt
```

---

### `calibrate.py` — Threshold optimiser

Sweeps a grid of `fraud_threshold × review_threshold` combinations to find the pair that maximises F1 subject to Recall ≥ 15%.

```
python calibrate.py [OPTIONS]

Options:
  --level TEXT              Competition level (required)
  --labels-file TEXT        Ground-truth label file (required)
  --split TEXT              Dataset split (default: train)
  --data-dir TEXT           Data directory
  --output-dir TEXT         Where to write thresholds JSON (default: thresholds)
  --max-transactions INT    Cap for smoke testing
  --fraud-grid TEXT         Comma-separated fraud threshold grid
                            (default: "0.50,0.55,0.60,0.65,0.70,0.75,0.80")
  --review-grid TEXT        Comma-separated review threshold grid
                            (default: "0.30,0.35,0.40,0.45,0.50,0.55")

Output:
  thresholds/{level}_thresholds.json  — best thresholds + metrics
```

**Example:**

```bash
python calibrate.py \
  --level brave-new-world \
  --labels-file outputs/brave-new-world_train_labels.txt \
  --fraud-grid "0.55,0.60,0.65,0.70" \
  --review-grid "0.35,0.40,0.45"
```

---

### `submit.py` — Competition submission runner

Runs inference on the **validation** split and flushes all Langfuse traces.

```
python submit.py [OPTIONS]

Options:
  --level TEXT              Competition level (required)
  --data-dir TEXT           Data directory
  --output-dir TEXT         Output directory (default: outputs)
  --max-transactions INT    Cap for smoke testing

Output:
  outputs/{level}_validation_output.txt  — submission file
  Langfuse session ID printed to stdout
```

**Example:**

```bash
python submit.py --level brave-new-world
```

---

## 12. Available Models

The pipeline routes LLM calls through **OpenRouter**, giving access to all major providers with a single API key. Set `FAST_MODEL` and `STRONG_MODEL` in `.env`.

### Recommended model combinations

| Use case | FAST_MODEL | STRONG_MODEL |
|---|---|---|
| **Default (balanced)** | `openai/gpt-4o-mini` | `openai/gpt-4o` |
| **Cost-optimised** | `openai/gpt-4o-mini` | `anthropic/claude-haiku-4-5` |
| **Maximum accuracy** | `openai/gpt-4o` | `anthropic/claude-opus-4-7` |
| **Claude-only** | `anthropic/claude-haiku-4-5` | `anthropic/claude-sonnet-4-6` |
| **Gemini** | `google/gemini-flash-1.5` | `google/gemini-2.5-pro` |
| **Local (Ollama)** | `ollama/llama3.1` | `ollama/llama3.1:70b` |

### OpenRouter model IDs

| Model | OpenRouter ID |
|---|---|
| GPT-4o | `openai/gpt-4o` |
| GPT-4o mini | `openai/gpt-4o-mini` |
| Claude Opus 4.7 | `anthropic/claude-opus-4-7` |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4-5` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini Flash 1.5 | `google/gemini-flash-1.5` |
| Llama 3.1 70B | `meta-llama/llama-3.1-70b-instruct` |

---

## 13. Evaluation & Metrics

### Metrics computed

| Metric | Formula | Competition role |
|---|---|---|
| **Precision** | TP / (TP + FP) | Quality of predictions |
| **Recall** | TP / (TP + FN) | **Must be ≥ 15% or disqualified** |
| **F1** | 2·P·R / (P + R) | Primary ranking metric |
| **FPR** | FP / (FP + TN) | Customer-experience impact |

### Label convention

- Output file includes `transaction_id` for all transactions with verdict **FRAUD** or **REVIEW**.
- **CLEAN** transactions are excluded from the output file.
- Ground-truth labels use the same convention (presence = positive).

### Interpreting results

```
Recall  < 0.15 → DISQUALIFIED
Recall  ≥ 0.15 → valid submission; higher F1 = better rank
Precision ~ 0.5 with Recall ~ 0.7 → strong result (industry baseline ≈ P=0.6, R=0.5)
```

---

## 14. Threshold Calibration

After running the pipeline on the training split, calibrate thresholds before final submission:

```bash
# Step 1: Run pipeline on training data (generates raw outputs)
python pipeline.py --level brave-new-world --split train

# Step 2: Calibrate thresholds with a fine grid
python calibrate.py \
  --level brave-new-world \
  --labels-file outputs/brave-new-world_train_labels.txt \
  --fraud-grid "0.50,0.55,0.60,0.65,0.70" \
  --review-grid "0.30,0.35,0.40,0.45,0.50"

# Step 3: Inspect output
cat thresholds/brave-new-world_thresholds.json
# {
#   "fraud_threshold": 0.60,
#   "review_threshold": 0.40,
#   "precision": 0.71,
#   "recall": 0.68,
#   "f1": 0.69,
#   "fpr": 0.12
# }
```

The calibrated thresholds are automatically loaded by `pipeline.py` on subsequent runs if `thresholds/{level}_thresholds.json` exists.

### Tuning the short-circuit threshold

```bash
# More aggressive short-circuit (fewer LLM calls, higher cost of missed fraud edge cases)
SHORTCIRCUIT_THRESHOLD=0.85 python pipeline.py --level brave-new-world --split train

# Conservative short-circuit (always run all agents except extreme cases)
SHORTCIRCUIT_THRESHOLD=0.95 python pipeline.py --level brave-new-world --split train
```

---

## 15. Submission

```bash
# Run on validation split (official submission)
python submit.py --level brave-new-world

# Check the output
cat outputs/brave-new-world_validation_output.txt

# Submit all three levels
for level in brave-new-world deus-ex the-truman-show; do
  python submit.py --level $level
done
```

The submission runner:
1. Runs the full pipeline on the **validation** split
2. Writes `outputs/{level}_validation_output.txt`
3. Flushes all pending Langfuse traces
4. Prints session ID and summary statistics

---

## 16. Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test module
pytest tests/test_featurizer.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test
pytest tests/test_agents.py::test_supervisor_verdict_clean -v
```

### Test modules

| Module | Coverage |
|---|---|
| `tests/test_state.py` | FraudState schema: all 48 keys, correct types, safe defaults |
| `tests/test_featurizer.py` | All 15 deterministic features; entity extraction; type correctness |
| `tests/test_agents.py` | JSON parsing + signal bounds for all 3 agents |
| `tests/test_graph.py` | Graph topology; short-circuit trigger; fan-out/fan-in |
| `tests/test_evaluate.py` | TP/FP/FN/TN computation; precision/recall/F1/FPR |

---

## 17. Observability with Langfuse

All LLM calls are automatically traced to the Langfuse dashboard configured in `.env`.

### Accessing traces

1. Open `https://challenges.reply.com/langfuse` in a browser
2. Sign in with hackathon credentials
3. Filter by session ID (printed at the start of each run)

### Session ID format

```
{TEAM_NAME}-{ULID}
# example: malto-team-3-01ARZ3NDEKTSV4RRFFQ69G5FAV
```

The session ID is printed to stdout at the start of every pipeline run and embedded in every trace.

### What is traced

| Event | Token count | Cost |
|---|---|---|
| Featurizer | 0 (no LLM) | \$0 |
| Transaction agent | ~800–1 200 tokens | ~\$0.001 |
| Comms agent | ~1 000–2 000 tokens | ~\$0.003 |
| Supervisor | ~600–1 000 tokens | ~\$0.002 |
| Short-circuit path | 0 specialist tokens | ~\$0.002 |

### Programmatic access

```python
from session import get_langfuse_client

lf = get_langfuse_client()
# lf is a langfuse.Langfuse instance
```

---

## 18. Performance Optimisations

### 1. Short-circuit routing (~75% LLM call reduction)

When `combined_risk_score ≥ SHORTCIRCUIT_THRESHOLD` (default 0.90), deterministic evidence is conclusive. Both specialist agents are skipped; the Supervisor processes the deterministic features directly.

### 2. Per-user statistics pre-computation

All per-sender statistics (velocity counts, amount distributions, balance history, PSI) are computed once per unique sender before the main transaction loop. This eliminates O(N × M) repeated DataFrame scans during inference.

### 3. Model routing

Expensive reasoning is concentrated where it matters:

- **Transaction agent** uses a cheaper, faster model (`gpt-4o-mini`) for structured feature synthesis.
- **Comms agent + Supervisor** use a stronger model (`gpt-4o`) for free-text reasoning and final arbitration.

### 4. Per-user communications filtering

SMS and email threads are pre-filtered to only those mentioning the sender's first or last name, preventing cross-user contamination in datasets where communications are pooled.

---

## 19. Project Structure

```
Multi_Agent_Fraud_Detection/
│
├── agents/
│   ├── __init__.py
│   ├── transaction_agent.py     # Node 2: LLM reasoning on transaction features
│   ├── comms_agent.py           # Node 3: LLM reasoning on communications
│   └── supervisor_agent.py      # Node 4: cost-aware arbitration
│
├── tools/
│   ├── __init__.py
│   ├── transaction_tools.py     # velocity, z-score, balance, IBAN, PSI
│   ├── geospatial_tools.py      # Haversine travel detection, location clustering
│   └── comms_tools.py           # regex entity extraction, mismatch detection
│
├── prompts/
│   ├── __init__.py
│   ├── transaction_agent_prompt.py  # 4-state SCoT + few-shot examples
│   ├── comms_agent_prompt.py        # 4-state SCoT + few-shot examples
│   └── supervisor_prompt.py         # cost-aware thresholds + decision rules
│
├── tests/
│   ├── __init__.py
│   ├── test_state.py
│   ├── test_featurizer.py
│   ├── test_agents.py
│   ├── test_graph.py
│   └── test_evaluate.py
│
├── outputs/                    # Pipeline output files (created on run)
├── train-validation/           # Competition data ZIPs (not committed)
├── Langfuse/                   # Observability example integration
│
├── data_loader.py              # ZIP loading + DataFrame normalisation
├── featurizer.py               # Node 1: deterministic feature computation
├── graph.py                    # LangGraph topology + graph compilation
├── state.py                    # FraudState TypedDict (48 fields)
├── session.py                  # Session ID + Langfuse client initialisation
│
├── pipeline.py                 # CLI: run inference on train/validation split
├── evaluate.py                 # CLI: compute precision/recall/F1/FPR
├── calibrate.py                # CLI: sweep threshold grid to maximise F1
├── submit.py                   # CLI: final validation submission + trace flush
│
├── requirements.txt
├── .env                        # Credentials + model routing (not committed)
└── README.md
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AuthenticationError` | Wrong `OPENROUTER_API_KEY` | Verify key at openrouter.ai/keys |
| `FileNotFoundError: transactions.csv` | Data not in `DATA_DIR` | Check `DATA_DIR` in `.env` and ZIP extraction |
| `Recall < 0.15` warning | Thresholds too strict | Run `calibrate.py` or lower `SHORTCIRCUIT_THRESHOLD` |
| Langfuse traces not appearing | Missing Langfuse keys or wrong host | Check `LANGFUSE_HOST` and credentials |
| All transactions flagged REVIEW | `combined_risk_score` normalisation issue | Inspect featurizer weight sum; should total ≤ 1.0 |
| JSON parse error from agent | LLM returned malformed JSON | Pipeline retries once; enable `--max-transactions 5` to debug |

---

## Team

**Malto-Team-3** — Reply Mirror Hackathon, April 2026

---

*Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://github.com/langchain-ai/langchain) · [Langfuse](https://langfuse.com)*
