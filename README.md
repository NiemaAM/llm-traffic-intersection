# 🚦 LLM-Driven Agents for Traffic Intersection Conflict Resolution

> **CSC5382 – AI for Digital Transformation**
> Production-ready ML system leveraging LLMs and generative AI for intelligent, safety-aware vehicle conflict resolution at urban intersections.

[![CI/CD](https://github.com/your-org/llm-traffic-intersection/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/your-org/llm-traffic-intersection/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Environment Setup (Ubuntu / WSL2)](#-environment-setup-ubuntu--wsl2)
- [Milestone 1: Project Inception](#milestone-1-project-inception)
- [Milestone 2: Proof of Concept](#milestone-2-proof-of-concept)
- [Milestone 3: Data Acquisition, Validation and Preparation](#milestone-3-data-acquisition-validation-and-preparation)
- [Milestone 4: Model Development and Evaluation](#milestone-4-model-development-and-evaluation)
- [Milestone 5: ML Productionization](#milestone-5-ml-productionization---api-development-packaging-deployment-and-serving)
- [Milestone 6: Model Testing, Evaluation, Monitoring and Continual Learning](#milestone-6-model-testing-evaluation-monitoring-and-continual-learning)

---

## 🎯 Project Overview

Urban intersections are high-risk zones where vehicle conflicts cause accidents, congestion, and economic losses. This project deploys a **Large Language Model (LLM)** as an intelligent intersection controller that detects potential vehicle conflicts, issues priority rankings and waiting-time decisions, provides human-readable natural language rationale for every decision, and operates within a full MLOps lifecycle: data → train → serve → monitor → retrain.

The system follows the **Human-in-the-Loop AI System** archetype: the LLM generates control decisions, engineers validate them, and iterative feedback improves performance over time.

---

## 🗂 Repository Structure

```
llm-traffic-intersection/
├── .github/workflows/       # CI/CD GitHub Actions
├── .dvc/                    # DVC configuration
├── configs/                 # Pipeline configs
├── data/
│   ├── raw/                 # Raw generated data
│   ├── interim/             # Intermediate data
│   ├── processed/           # Feature-engineered data
│   ├── external/            # Intersection layout JSON
│   └── feature_store/       # Feast feature store
├── deployment/
│   ├── docker/              # Dockerfile, docker-compose.yml, prometheus.yml
│   └── k8s/                 # Kubernetes manifests
├── docs/                    # Architecture diagrams
├── model/
│   └── model_card.md        # Hugging Face–style model card
├── models/                  # Saved scaler and artifacts
├── notebooks/
│   ├── milestone3_data_pipeline.ipynb
│   └── milestone4_model_development.ipynb
├── references/
│   └── references.bib       # BibTeX references
├── reports/                 # Evaluation reports, drift reports
├── scripts/
│   └── run_all_pipelines.py # Master pipeline runner
├── src/
│   ├── poc/
│   │   ├── conflict_detection_orig.py  # Original rule-based engine (Masri et al.)
│   │   ├── visualization_orig.py       # Animated Plotly visualization
│   │   ├── data_generation.py          # Original data generator
│   │   ├── utils.py                    # Utility functions
│   │   └── poc_app.py                  # Milestone 2 Streamlit PoC
│   ├── data/
│   │   ├── generate_data.py            # Synthetic data generation
│   │   └── validate_data.py            # Schema + Great Expectations
│   ├── features/
│   │   └── preprocess.py               # scikit-learn feature pipeline
│   ├── models/
│   │   ├── llm_model.py                # LLM inference + prompts + fine-tuning
│   │   └── train.py                    # MLflow experiment tracking
│   ├── evaluation/
│   │   └── evaluate.py                 # Accuracy, robustness, bias audit
│   ├── api/
│   │   ├── app.py                      # FastAPI serving endpoint
│   │   └── streamlit_app.py            # Milestone 5 production Streamlit
│   ├── monitoring/
│   │   └── monitor.py                  # Drift, Prometheus, A/B, CT/CD
│   └── pipelines/
│       ├── data_pipeline.py            # ZenML M3 data pipeline
│       ├── training_pipeline.py        # ZenML M4 training pipeline
│       ├── serving_pipeline.py         # ZenML M5 serving pipeline
│       └── monitoring_pipeline.py      # ZenML M6 monitoring pipeline
└── tests/
    ├── unit/
    └── integration/
```

---

## 🛠 Environment Setup (Ubuntu / WSL2)

This section documents the exact steps followed to set up the project on **Ubuntu 22.04 running under WSL2 on Windows**.

### Prerequisites

> **Use WSL2, not WSL1.** Verify with `wsl --list --verbose` in PowerShell. WSL2 has a real Linux kernel required for DVC, ZenML, and Docker.
> Keep all project files **inside** the WSL filesystem (`~/`) — not under `/mnt/c/`. Accessing Windows files from WSL is slower and causes issues with DVC checksumming.

### Step 1 — Update Ubuntu and install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl wget unzip build-essential
python3.11 --version   # should print Python 3.11.x
```

### Step 2 — Create and activate virtual environment

```bash
cd ~/llm-traffic-intersection
python3.11 -m venv .venv
source .venv/bin/activate
# Your prompt will show (.venv)
# To reactivate after closing terminal:
# source ~/llm-traffic-intersection/.venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses `"zenml[server]>=0.60.0"` (with quotes) to install ZenML with all dashboard dependencies. The base `zenml` package alone causes `ModuleNotFoundError: No module named 'sqlalchemy_utils'` when running `zenml init` or `zenml login --local`.

### Step 4 — Configure environment variables

```bash
cp .env.example .env
nano .env
```

Set your values. **Important `.env` formatting rules** — inline comments break parsing:

```bash
# ✅ CORRECT — comment on its own line
# Optional fine-tuned model ID
FINE_TUNED_MODEL_ID=

# ❌ WRONG — comment on same line becomes part of the value
FINE_TUNED_MODEL_ID=     # Optional: your fine-tuned model ID (ft:gpt-4o-mini:...)
```

Correct `.env` content:

```bash
OPENAI_API_KEY=sk-proj-...your-full-key...

# Optional: set after fine-tuning completes
FINE_TUNED_MODEL_ID=

MODEL_NAME=gpt-4o-mini
FEW_SHOT=true

MLFLOW_TRACKING_URI=http://localhost:5000
API_URL=http://localhost:8000
PORT=8000

HF_TOKEN=hf_...
HF_SPACE=your-username/traffic-intersection

GRAFANA_PASSWORD=admin
```

### Step 5 — Initialize Git

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git init
git add .
git commit -m "Initial commit"
```

### Step 6 — Initialize DVC

#### Fix required: `.gitignore` blocks `.dvc/` folder

The default `.gitignore` had a `*.dvc` pattern that blocked DVC from initializing. Fix it:

```bash
nano .gitignore
```

Change the DVC section from:
```
# DVC
*.dvc
!.dvc/config
```

To:
```
# DVC
!.dvc/
!.dvc/config
```

Then initialize DVC:

```bash
dvc init -f       # -f forces re-init if .dvc/ already partially created
git add .dvc .dvcignore
git commit -m "Initialize DVC"

dvc add data/raw/generated_dataset.csv
dvc add data/processed/features.csv
git add data/raw/generated_dataset.csv.dvc data/processed/features.csv.dvc .gitignore
git commit -m "Track data with DVC"
```

### Step 7 — Initialize ZenML

```bash
zenml init
```

If you see `ERROR: failed to initiate DVC - /home/.../.dvc is ignored`, it means the `.gitignore` fix above was not applied yet.

#### Start the ZenML dashboard (optional)

```bash
pip install "zenml[server]"    # if not already installed via requirements.txt
zenml login --local
# Opens dashboard at http://127.0.0.1:8237
# Login: username=default, password=(empty)
```

> **Note:** `zenml up` is deprecated in ZenML 0.94+. Use `zenml login --local` instead.
> The dashboard runs in the foreground — open a second terminal tab for other commands.

### Step 8 — Install Docker (via Docker Desktop for Windows)

Docker Desktop is the recommended approach for WSL2. The `get.docker.com` script shows a 20-second warning and may fail with package conflicts.

1. Download **Docker Desktop for Windows** from https://www.docker.com/products/docker-desktop/
2. During install, ensure **"Use WSL 2 based engine"** is checked
3. After install: **Settings → Resources → WSL Integration → Enable Ubuntu-22.04**
4. Verify inside Ubuntu terminal:

```bash
docker --version          # Docker version 28.x.x
docker compose version    # Docker Compose version v2.x.x
```

### Step 9 — Run tests to verify setup

```bash
PYTHONPATH=. pytest tests/ -v
# Expected: 63 passed
```

---

## ⚡ Quick Start

```bash
# After setup above:
source .venv/bin/activate

# Milestone 2 PoC (rule-based, no API key needed)
PYTHONPATH=. streamlit run src/poc/poc_app.py --server.port 8503

# Run all pipelines (M3→M6)
PYTHONPATH=. python scripts/run_all_pipelines.py

# Production API
PYTHONPATH=. uvicorn src.api.app:app --reload --port 8000

# Full Docker stack (M5)
cd deployment/docker && docker compose up
```

---

---

## Milestone 1: Project Inception

### Introduction

This milestone frames the traffic intersection conflict problem as a production ML task. It identifies the business case and its measurable value, selects and justifies the baseline model with full reproducibility, defines the project archetype, and establishes the evaluation metrics that will govern all subsequent milestones.

### Goal of this Milestone

Define the *what*, *why*, and *how* of the project with enough rigour that the problem is solvable, the baseline is reproducible, and success criteria are measurable.

### References

| # | Citation | Link |
|---|---|---|
| [1] | Masri, Ashqar, Elhenawy (2025) — LLMs as Traffic Control Systems | [arXiv:2411.10869](https://arxiv.org/abs/2411.10869) · [GitHub](https://github.com/sarimasri3/Intersection-Conflict-Detection) |
| [2] | Li et al. (2025) — LLM-TrafficBrain | arXiv |
| [3] | Lai et al. (2025) — LLMLight / LightGPT | arXiv |
| [4] | Wang et al. (2025) — LLM-DCTSC | arXiv |

Full BibTeX: [`references/references.bib`](references/references.bib)

### Steps Followed

1. Identified the urban traffic congestion problem; scoped it to 4-way intersection conflict detection.
2. Reviewed 2024–2025 literature on LLMs for traffic signal control.
3. Selected Masri et al. (2025) as the baseline — the only paper providing both trained model and retrain notebook.
4. Wrote the model card in Hugging Face format.
5. Defined safety-oriented (recall, FNR) and operational (AWT, AQL) evaluation metrics.
6. Documented the Human-in-the-Loop project archetype.

### Model Card Link

→ [`model/model_card.md`](model/model_card.md)

---

### 1.1 Framing the Business Idea as an ML Problem

#### Business Case Description

Urban congestion costs cities billions of dollars annually in lost productivity, fuel waste, and accident costs. Traditional traffic signal controllers rely on fixed timing plans or rule-based adaptive logic that cannot respond dynamically to real-time vehicle distributions or edge cases such as emergency vehicles.

This project proposes an **LLM-driven intersection controller** that analyzes each approaching vehicle's speed, distance, lane, and intended direction; identifies vehicles on a collision course; and issues priority rankings with natural language control decisions. The system operates within a Human-in-the-Loop validation loop. It targets **urban planners, municipalities, and smart city operators**.

#### Business Value of Using ML

| Category | Value Delivered |
|---|---|
| 🚦 **Operational** | Reduced waiting times; higher intersection throughput; fewer bottlenecks |
| 🌳 **Environmental** | Shorter idle times → lower CO₂ and NOₓ emissions per trip |
| 💰 **Economic** | Reduced fuel consumption; lower accident costs; fewer manual re-configurations |
| 🏙️ **Strategic** | Scales city-wide; adapts to novel traffic patterns without re-engineering rules |
| 🔍 **Explainability** | Every decision includes a natural language rationale, supporting accountability |

#### Data Overview

**Source:** [Zenodo DOI: 10.5281/zenodo.14171745](https://doi.org/10.5281/zenodo.14171745)

Simulated multi-lane intersection scenarios generated by `src/data/generate_data.py`. Each record represents one vehicle within one scenario.

**Input features (per vehicle):**

| Feature | Description | Example |
|---|---|---|
| `vehicle_id` | Unique identifier | `V7657` |
| `lane` | Lane number (1–8) | `1` |
| `speed` | Vehicle speed (km/h) | `62.36` |
| `distance_to_intersection` | Distance in metres | `45.0` |
| `direction` | Cardinal direction | `north` |
| `destination` | Intended exit | `F` |

**Output labels (per scenario):**

| Label | Description | Example |
|---|---|---|
| `is_conflict` | Conflict present? | `yes` |
| `number_of_conflicts` | Count of conflict pairs | `1` |
| `conflict_vehicles` | Vehicle ID pairs in conflict | `[{"vehicle1_id":"V001","vehicle2_id":"V002"}]` |
| `decisions` | Natural language control actions | `["V002 must yield to V001"]` |
| `priority_order` | Per-vehicle priority rank | `{"V001":1,"V002":2}` |
| `waiting_times` | Per-vehicle wait in seconds | `{"V001":0,"V002":3}` |

#### Project Archetype

**Human-in-the-Loop AI Decision Support System:**

```
Sensor / Simulation Data
        ↓
  LLM Conflict Analyzer   ←── prompt engineering / fine-tuning
        ↓
  Control Decisions        (priority, wait times, decisions)
        ↓
  Human Engineer Validates
        ↓
  Traffic Infrastructure   (signal phase changes)
        ↑
  Performance Feedback     (loop back to LLM improvement)
```

---

### 1.2 Feasibility Analysis

#### Literature Review

| Paper | Key Contribution | Conflict Detection | CoT Reasoning | Fine-tuned |
|---|---|---|---|---|
| **Masri et al. [1]** — *our baseline* | 4D system; GPT-4o-mini fine-tuned | ✅ | ✅ | ✅ |
| **LLM-TrafficBrain [2]** | Semantic reasoning with SUMO | — | ◐ | — |
| **LLMLight [3]** | First LLM as direct TSC agent | — | ✅ | ✅ |
| **LLM-DCTSC [4]** | DPO + RL reward; joint phase+duration | — | ✅ | ✅ |

#### Model Choice / Specification of a Baseline

**Baseline:** Fine-tuned `GPT-4o-mini` — Masri et al. (2025)

Arguments: published ~83% accuracy benchmark, full reproducibility via GitHub repo, task alignment with structured JSON output, natural language explainability.

| Model | Mode | Accuracy |
|---|---|---|
| **GPT-mini (fine-tuned)** | Fine-tuned | **~83%** |
| GPT-4o-mini (fine-tuned) | Fine-tuned | ~81% |
| GPT-4o-mini | Few-shot | ~71% |
| GPT-mini | Zero-shot | ~62% |
| LLaMA-3.1 (fine-tuned) | Fine-tuned | ~51% |

#### Metrics for Business Goal Evaluation

**Safety metrics (primary):**

| Metric | Business Significance |
|---|---|
| **Recall** | Minimises missed conflicts → prevents accidents |
| **F1-Score** | Balanced safety/efficiency tradeoff |
| **FNR** | Directly quantifies collision risk |
| **Precision** | Minimises false alarms |

**Operational metrics (secondary):** Average Waiting Time (AWT), Average Queue Length (AQL), Intersection Throughput, Average Travel Time (ATT).

---

---

## Milestone 2: Proof of Concept

### Introduction

A functional, interactive PoC demonstrating that LLM-driven conflict detection works end-to-end with a Streamlit app, animated visualizations, and a dual-engine selector (rule-based + LLM).

### Goal of this Milestone

Validate the technical approach before investing in full MLOps infrastructure.

### Tools Used

| Tool | Purpose | Version |
|---|---|---|
| **Streamlit** | Interactive web front-end | `>=1.35` |
| **Plotly** | Animated intersection visualization | `>=5.20` |
| **OpenAI API** | LLM inference (optional) | `>=1.30` |
| **pandas** | Tabular results display | `>=2.1` |
| **pytest** | Automated scenario testing | `>=8.0` |

### Steps Followed

1. Implemented rule-based conflict detection engine (`src/poc/conflict_detection_orig.py`) — exact Masri et al. logic.
2. Implemented animated Plotly visualization (`src/poc/visualization_orig.py`) — problem + solution views.
3. Built Streamlit PoC app with vehicle editor, preset scenarios, dual-engine selector, live JSON panel.
4. Added auto-loading of API key from `.env` (no manual input required).
5. Added dual-engine test runner (rule-based / LLM / both compare).

### How to Use

```bash
# No API key needed for rule-based mode
PYTHONPATH=. streamlit run src/poc/poc_app.py --server.port 8503
# Open: http://localhost:8503
```

---

### 2.1 Model Integration

Two engines selectable from the sidebar:

**Engine A — Rule-Based** (`src/poc/conflict_detection_orig.py`): deterministic, offline, instant. Uses `paths_cross()`, `arrival_time_close()`, `apply_priority_rules()` from Masri et al.

**Engine B — LLM** (`src/models/llm_model.py`): GPT-4o-mini with system prompt + few-shot examples. API key and model name loaded automatically from `.env`.

---

### 2.2 App Development

Sidebar: engine selector, API key status (auto-loaded from `.env`), layout reference, animation controls.

Main: preset loader (4 named scenarios), random generator, interactive vehicle editor, live bidirectional JSON panel.

Results: conflict banner, KPI metrics, decisions list, priority table, animated problem view, animated solution view.

Tests: 3-way engine selector (rule-based / LLM / both), pass/fail table with agreement column.

---

### 2.3 End-to-End Scenario Testing

Five hand-designed scenarios verify all key behaviours. All run offline (rule-based) or against the LLM API.

| # | Scenario | Expected |
|---|---|---|
| 1 | N(lane1)↔E(lane3), same speed & distance | Conflict |
| 2 | Arrival time gap too large | No conflict |
| 3 | Same direction (N lane1 & N lane2) | No conflict |
| 4 | S(lane5)↔W(lane7), close arrival | Conflict |
| 5 | Three vehicles, close arrival | Conflict |

---

---

## Milestone 3: Data Acquisition, Validation and Preparation

### Introduction

Full data lifecycle: ingestion, validation, feature engineering, versioning, and feature store — orchestrated as a ZenML pipeline.

### Goal of this Milestone

Build a production-grade, reproducible data pipeline ensuring data quality, traceability, and feature reuse.

### Tools Used

| Tool | Purpose | Version |
|---|---|---|
| **Great Expectations** | Data validation and schema enforcement | `>=0.18` |
| **scikit-learn Pipeline** | Feature engineering | `>=1.4` |
| **DVC** | Data versioning | `>=3.40` |
| **Feast** | Feature store | `>=0.38` |
| **ZenML** | Pipeline orchestration | `>=0.60` |

### Steps Followed

1. Schema definition — 14 typed columns with constraints.
2. Data generation — parametric synthetic scenario generator.
3. Validation — pandas validator + Great Expectations HTML report.
4. Feature engineering — 6 custom scikit-learn transformers.
5. DVC versioning — initialized after fixing `.gitignore` (see [Environment Setup](#-environment-setup-ubuntu--wsl2)).
6. Feast feature store — local SQLite provider.
7. ZenML pipeline integration.

### How to Use

```bash
# Notebook (interactive)
jupyter notebook notebooks/milestone3_data_pipeline.ipynb

# Pipeline only
PYTHONPATH=. python src/pipelines/data_pipeline.py

# Or via master runner
PYTHONPATH=. python scripts/run_all_pipelines.py --milestone 3
```

---

### 3.1 Schema Definition

14-column schema in `src/data/validate_data.py`. Key constraints: `direction` ∈ {north,south,east,west}, `is_conflict` ∈ {yes,no}, `speed` ∈ [0,200], `lane` ∈ [1,10].

---

### 3.2 Data Validation and Verification

Two layers: pandas `validate_schema()` (always available) + Great Expectations suite (HTML Data Docs report). The ZenML `validate_data` step raises `ValueError` on failure, halting the pipeline.

---

### 3.3 Data Versioning

DVC tracks `data/raw/generated_dataset.csv`, `data/processed/features.csv`, `models/scaler.joblib`. The `version_data` ZenML step calls `dvc add` after each run.

**Fix applied:** Modified `.gitignore` to change `*.dvc` → `!.dvc/` so the `.dvc/` directory is not ignored by Git (required for DVC to initialize).

---

### 3.4 Setting Up a Feature Store

Feast with local SQLite provider. Two feature views: `vehicle_intersection_features` and `scenario_aggregate_features`, combined in `conflict_detection_service`.

---

### 3.5 Setup of Data Pipeline within the Larger ML Pipeline / MLOps Platform

ZenML pipeline `m3_data_pipeline`:
```
ingest_data → validate_data → engineer_features → version_data → push_to_feature_store
```

Feature transformers: `DirectionEncoder`, `ConflictFlagEncoder`, `WaitingTimeExtractor`, `PriorityExtractor`, `ScenarioAggFeatures`, `DropRawColumns` + `StandardScaler`.

---

---

## Milestone 4: Model Development and Evaluation

### Introduction

Full LLM training pipeline: prompt engineering, fine-tuning via OpenAI API, experiment tracking with MLflow, all orchestrated as a ZenML pipeline with 7 steps.

### Goal of this Milestone

Define, train, evaluate, and version the LLM conflict detector with reproducible experiments tracked in MLflow.

### Tools Used

| Tool | Purpose | Version |
|---|---|---|
| **OpenAI API** | LLM inference and fine-tuning | `>=1.30` |
| **MLflow** | Experiment tracking and model versioning | `>=2.12` |
| **ZenML** | Training pipeline orchestration | `>=0.60` |
| **scikit-learn** | Evaluation metrics | `>=1.4` |
| **CodeCarbon** | Energy efficiency measurement | `>=2.4` |

### Steps Followed

1. Implemented `IntersectionLLM` wrapper (zero-shot, few-shot, fine-tuned modes).
2. Designed system prompt, few-shot examples, JSON output schema.
3. Built JSONL fine-tuning dataset export.
4. Integrated MLflow for all experiment tracking.
5. Built 7-step ZenML training pipeline including real `train_model` step.
6. Added CodeCarbon energy measurement.

### How to Use

```bash
# Notebook (interactive, recommended)
jupyter notebook notebooks/milestone4_model_development.ipynb

# Pipeline (skip fine-tuning, evaluation only)
PYTHONPATH=. python scripts/run_all_pipelines.py --milestone 4

# Pipeline with fine-tuning enabled (~$1-2, 10-30 min)
PYTHONPATH=. python scripts/run_all_pipelines.py --milestone 4 --train

# View results
mlflow ui --port 5000   # or open http://localhost:5000 if Docker stack running
```

---

### 4.1 Project Structure Definition and Modularity

Cookiecutter Data Science layout. Single-responsibility modules: `llm_model.py` for inference, `train.py` for orchestration, `pipelines/` for ZenML steps only.

---

### 4.2 Code Versioning

GitHub Flow: `main` ← `develop` ← feature branches. All PRs gated by CI. Releases tagged `v1.0` through `v6.0`.

---

### 4.3 Experiment Tracking and Model Versioning

ZenML pipeline `m4_training_pipeline` (7 steps):

```
prepare_finetune_data
        │
        ├──► evaluate_baseline  (zero-shot)   ──┐
        │                                        │
        ├──► evaluate_few_shot  (few-shot)    ──►├──► compare_and_register ──► measure_energy
        │                                        │
        └──► train_model ──► evaluate_finetuned ─┘
```

The `train_model` step is the **`model.fit()` equivalent** for LLMs: uploads the JSONL file, starts the OpenAI fine-tuning job, polls every 30 seconds until complete, and returns the resulting model ID.

MLflow runs logged:

| Run | Params | Metrics |
|---|---|---|
| `zero-shot-baseline` | model, few_shot=False | accuracy, F1, recall, FNR, latency |
| `few-shot-eval` | model, few_shot=True | accuracy, F1, recall, FNR, latency |
| `fine-tuning-job` | base_model, n_epochs, job_id | training_duration_s |
| `fine-tuned-eval` | fine_tuned_model_id | accuracy, F1, recall, FNR, latency |
| `model-comparison` | best_config, best_f1 | all variant metrics |
| `energy-measurement` | best_config | co2_kg, co2_g |

---

### 4.4 Integration of Model Training and Offline Evaluation

The `compare_and_register` step picks the best variant by F1 score, logs a comparison table to MLflow as a JSON artifact (`reports/model_comparison.json`), and records the fine-tuned model ID for use in Milestone 5.

---

### 4.5 Energy Efficiency Measurement

CodeCarbon tracks CO₂ emissions per pipeline run. Results logged to MLflow `energy-measurement` run as `co2_kg` and `co2_g` metrics.

---

---

## Milestone 5: ML Productionization - API Development, Packaging, Deployment and Serving

### Introduction

Packages the LLM conflict resolver into a production-ready containerised web service with automated CI/CD.

### Goal of this Milestone

Ship the model as a reliable, containerised, monitored web service accessible via API and Streamlit UI.

### Tools Used

| Tool | Purpose | Version |
|---|---|---|
| **FastAPI** | REST API framework | `>=0.111` |
| **Uvicorn** | ASGI server | `>=0.29` |
| **Streamlit** | Front-end client | `>=1.35` |
| **Docker + Compose** | Containerisation | Latest |
| **GitHub Actions** | CI/CD pipeline | — |
| **Hugging Face Spaces** | Cloud hosting | — |
| **Prometheus + Grafana** | Infrastructure monitoring | Latest |

### Steps Followed

1. Implemented FastAPI with `/predict`, `/predict/batch`, `/health`.
2. Built production Streamlit front-end (same style as PoC, calls FastAPI).
3. Multi-stage Dockerfile (builder + slim production image, non-root user).
4. docker-compose.yml with 5 services.
5. GitHub Actions CI/CD.
6. ZenML serving pipeline for API validation and test registration.

### How to Use

```bash
# Local development
PYTHONPATH=. uvicorn src.api.app:app --reload --port 8000
PYTHONPATH=. streamlit run src/api/streamlit_app.py --server.port 8501

# Docker Compose (full stack)
cd deployment/docker
# Fix: remove inline comments from .env before running
docker compose up --build

# Ports (fix conflicts by editing docker-compose.yml):
#   API:        http://localhost:8000  (docs: /docs)
#   Streamlit:  http://localhost:8501
#   MLflow:     http://localhost:5000
#   Grafana:    http://localhost:3001  (login: admin/admin)
#   Prometheus: http://localhost:9091

# Run ZenML serving pipeline
PYTHONPATH=. python scripts/run_all_pipelines.py --milestone 5
```

**Port conflict fixes applied during setup:**
- Port 3000 was occupied by a pre-existing Grafana container → changed to `3001:3000`
- Port 9090 was occupied → changed to `9091:9090`

---

### 5.1 ML System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                         │
│   Streamlit UI (8501) ────────  REST Clients            │
└──────────────────────┬────────────────┬─────────────────┘
                       │ HTTP           │ HTTP
                       ▼                ▼
┌─────────────────────────────────────────────────────────┐
│   FastAPI (8000) + Uvicorn (2 workers)                  │
│   POST /predict  ·  POST /predict/batch  ·  GET /health │
│              ↓                                          │
│       IntersectionLLM  (few-shot / fine-tuned)         │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTPS
                        ▼
              ┌─────────────────┐
              │  OpenAI API     │
              │  gpt-4o-mini    │
              └─────────────────┘
┌─────────────────────────────────────────────────────────┐
│  MLOps: MLflow (5000) · Prometheus (9091) · Grafana (3001) · ZenML │
└─────────────────────────────────────────────────────────┘
```

**Serving mode:** On-demand (request-response) at `/predict`; batch at `/predict/batch`.

---

### 5.2 Application Development

**FastAPI** (`src/api/app.py`): Singleton `IntersectionLLM` via `get_llm()`, Pydantic V2 request/response validation, latency measurement, batch endpoint.

**Streamlit front-end** (`src/api/streamlit_app.py`): Same look and feel as the Milestone 2 PoC — preset scenarios, vehicle editor, live JSON panel, conflict results dashboard, plus a live API test runner that calls `/predict` and `/health` directly.

**Key difference from M2 PoC:** The production Streamlit app calls the **FastAPI service** over HTTP rather than running inference locally. This means it works even from a mobile browser or a different machine.

---

### 5.3 Integration and Deployment

**Multi-stage Dockerfile** → non-root user (`appuser`), health check via HTTP probe, minimal image size.

**docker-compose.yml fixes applied:**
```yaml
# Fix port conflicts (change left-side port only; right-side stays internal)
prometheus:
  ports:
    - "9091:9090"    # was 9090:9090 — conflicted with existing prometheus container
grafana:
  ports:
    - "3001:3000"    # was 3000:3000 — conflicted with existing grafana container

# Fix: remove "version:" key (obsolete in Compose v2, causes warning)
# Remove the line: version: "3.9"
```

**CI/CD Pipeline** (`.github/workflows/ci_cd.yml`): lint → tests → data pipeline → Docker push → HF deploy.

---

### 5.4 Model Serving

API key and model name are injected via environment variables (`OPENAI_API_KEY`, `MODEL_NAME`) set in `docker-compose.yml` from the `.env` file. The `IntersectionLLM` singleton is created once at startup and reused across requests.

**Troubleshooting:** If the API returns `invalid model ID`, check:
1. `FINE_TUNED_MODEL_ID` in `.env` must be **completely empty** (no inline comment)
2. Test the key directly: `docker exec traffic_api python3 -c "import openai, os; client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY']); print(client.models.list().data[0].id)"`
3. If you see `429 insufficient_quota` → add credits at https://platform.openai.com/settings/billing

---

---

## Milestone 6: Model Testing, Evaluation, Monitoring and Continual Learning

### Introduction

Closes the ML lifecycle loop with testing beyond accuracy, drift detection, monitoring dashboards, and automated continual learning triggers.

### Goal of this Milestone

Ensure the deployed model remains safe, unbiased, and performant over time.

### Tools Used

| Tool | Purpose | Version |
|---|---|---|
| **pytest** | Unit, integration, behavioral tests | `>=8.0` |
| **Evidently** | Data drift and model performance monitoring | `>=0.4.30` |
| **WhyLogs** | Prediction logging and data profiling | `>=1.4` |
| **Prometheus** | Infrastructure metrics | Latest |
| **Grafana** | Monitoring dashboards | Latest |
| **MLflow** | Monitoring run logging | `>=2.12` |
| **ZenML** | Monitoring pipeline orchestration | `>=0.60` |

### Steps Followed

1. Built `evaluate_model()` for held-out test evaluation.
2. Implemented `ABTestRouter` for online A/B testing.
3. Implemented 4 robustness / adversarial tests.
4. Implemented bias audit across direction groups.
5. Built drift detection with Evidently + statistical fallback.
6. Built prediction logger with WhyLogs + JSON fallback.
7. Built Prometheus metrics (counter, histogram, gauge).
8. Implemented `ContinualLearningTrigger`.
9. Assembled 7-step ZenML monitoring pipeline.

### How to Use

```bash
# Run monitoring pipeline
PYTHONPATH=. python scripts/run_all_pipelines.py --milestone 6

# Run all pipelines
PYTHONPATH=. python scripts/run_all_pipelines.py

# Quick mode (fewer API calls)
PYTHONPATH=. python scripts/run_all_pipelines.py --quick

# View drift report
open reports/drift_report.html
```

---

### 6.1 Model Evaluation and Testing

**Test set:** `evaluate_model()` — groups scenarios by `scenario_id`, computes accuracy, precision, recall, F1, FNR.

**A/B Testing** (`ABTestRouter`): deterministic routing by `request_id` hash. Records per-variant outcomes. Supports SciPy `chi2_contingency` significance test.

---

### 6.2 Testing Beyond Accuracy

**Bias Audit** (`audit_bias`): per-direction F1 and conflict rate. Flags bias if F1 disparity > 0.15.

**Robustness Tests** (`RobustnessTests`):

| Test | Requirement |
|---|---|
| `test_obvious_conflict` | Must predict conflict for close N↔S vehicles |
| `test_no_conflict_far` | Must predict no conflict when vehicles are 500+ m away |
| `test_perturbation_stability` | ±2 km/h speed change must not flip prediction |
| `test_priority_consistency` | Faster vehicle must always get rank 1 |

**Explainability:** Every prediction includes natural language `decisions` — intrinsically interpretable.

---

### 6.3 Model Monitoring and Continual Learning

**Performance monitoring:** Prometheus counters + Grafana dashboards.

**Drift monitoring:** Evidently `DataDriftPreset` with HTML report. Statistical fallback (mean/std comparison).

**CT/CD triggers** (`ContinualLearningTrigger`):

| Condition | Threshold | Action |
|---|---|---|
| F1 drops below threshold | 0.70 | Trigger retraining |
| Data drift detected | Evidently flag | Trigger retraining |
| Robustness pass rate low | < 75% | Trigger retraining |

**ZenML monitoring pipeline** `m6_monitoring_pipeline`:
```
load_reference_data + load_production_data → detect_drift
evaluate_on_test_set + run_robustness_tests + audit_model_bias
        → continual_learning_decision → log_monitoring_results (MLflow)
```

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 References

- [1] Masri, Ashqar, Elhenawy (2025) — [arXiv:2411.10869](https://arxiv.org/abs/2411.10869) | [GitHub](https://github.com/sarimasri3/Intersection-Conflict-Detection)
- [2] Li et al. (2025) — LLM-TrafficBrain
- [3] Lai et al. (2025) — LLMLight / LightGPT
- [4] Wang et al. (2025) — LLM-DCTSC
- Original project: [NiemaAM/LLM-Driven-Agents-for-Traffic-Signal-Optimization](https://github.com/NiemaAM/LLM-Driven-Agents-for-Traffic-Signal-Optimization)
- M4 Inspiration: [Ayman-AITACHOUR/PseudoCodeRAG-Translator](https://github.com/Ayman-AITACHOUR/PseudoCodeRAG-Translator)

Full BibTeX: [`references/references.bib`](references/references.bib)
