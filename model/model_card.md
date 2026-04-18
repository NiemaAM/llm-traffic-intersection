# 🚦 Model Card: Intersection Conflict Detection LLM

**Model ID:** `ft:gpt-4o-mini-2024-07-18:org::intersection-conflict`
**Base Model:** `gpt-4o-mini` (OpenAI)
**Task:** Multi-label structured classification — Intersection Conflict Detection & Vehicle Priority Assignment
**Milestone:** 1 – Project Inception (Baseline Specification)

---

## Model Description

This model is a fine-tuned version of **GPT-4o-mini** for the task of detecting vehicle conflicts at 4-way urban intersections and issuing structured control decisions. It is the **baseline model** for the LLM-Driven Agents for Traffic Intersection Conflict Resolution project.

The model receives a JSON description of 2–8 vehicles approaching an intersection (with speed, distance, lane, and direction) and outputs:
- Whether a conflict exists (`yes` / `no`)
- The number and location of conflicts
- Which vehicle pairs are in conflict
- Control decisions (which vehicle must yield)
- Priority rankings and estimated waiting times

### Model Type
Fine-tuned Large Language Model (Decoder-only, GPT family) with structured JSON output via prompt engineering and supervised fine-tuning.

### Languages
English (system prompts and decisions are in English)

---

## Intended Uses

### ✅ Direct Use
- Traffic signal decision support at urban intersections
- Smart city infrastructure advisory systems
- Autonomous vehicle coordination systems (advisory layer)
- Simulation-based traffic research and benchmarking

### ⚠️ Out-of-Scope Use
- Direct physical control of traffic infrastructure without human validation
- Deployment in safety-critical systems without an independent validation layer
- Use with real-time sensor data without re-training on real-world distributions

---

## Training Data

| Property | Value |
|---|---|
| **Source** | Synthetic intersection scenarios — [Zenodo dataset](https://doi.org/10.5281/zenodo.14171745) |
| **Generator** | `src/data/generate_data.py` (parametric scenario synthesis) |
| **Size** | ~1,000 vehicle records / ~200–400 unique scenarios |
| **Format** | OpenAI fine-tuning JSONL (system + user + assistant messages) |
| **Train/Val split** | 80% / 20% |
| **Vehicles per scenario** | 2–8 (randomized) |

### Input features per vehicle

| Feature | Type | Range |
|---|---|---|
| `vehicle_id` | string | V0001–V9999 |
| `lane` | int | 1–2 |
| `speed` | float (km/h) | 10–90 |
| `distance_to_intersection` | float (m) | 10–500 |
| `direction` | enum | north, south, east, west |
| `destination` | enum | A–H |

### Output labels

| Label | Type |
|---|---|
| `is_conflict` | binary: yes/no |
| `number_of_conflicts` | int ≥ 0 |
| `conflict_vehicles` | list of vehicle ID pairs |
| `decisions` | list of natural language control instructions |
| `priority_order` | dict: vehicle_id → priority rank (1 = highest) |
| `waiting_times` | dict: vehicle_id → wait seconds |

---

## Evaluation Results

Results from Masri et al. (2025) on the held-out test set (mixed vehicle count scenarios):

| Model Variant | Setting | Accuracy | F1 |
|---|---|---|---|
| **GPT-mini (fine-tuned)** | Mixed vehicle count | **~83%** | **~0.81** |
| GPT-4o-mini (fine-tuned) | 4 vehicles fixed | ~81% | ~0.79 |
| GPT-4o-mini (fine-tuned) | 8 vehicles fixed | ~71% | ~0.68 |
| GPT-mini (zero-shot) | Mixed scenarios | ~62% | ~0.58 |
| Meta-LLaMA-3.1 variants | Fine-tuned | ~51% | ~0.48 |
| Gemini (fine-tuned) | Various | Lower | — |

### Business Metrics

| Metric | Value | Significance |
|---|---|---|
| **Recall (Conflict)** | ~0.85 | Critical — minimises missed conflicts (safety risk) |
| **Precision** | ~0.79 | Minimises false alarms (unnecessary stops) |
| **FNR** | ~0.15 | False negative rate — directly linked to accident risk |
| **Avg. latency** | ~1.2 s | Acceptable for near-real-time advisory |

---

## Model Architecture & Training

- **Base:** GPT-4o-mini (OpenAI decoder-only transformer)
- **Fine-tuning method:** Supervised fine-tuning via OpenAI fine-tuning API
- **Hyperparameters:** n_epochs=3, learning_rate_multiplier=auto, batch_size=auto
- **Output format:** Forced JSON via `response_format={"type": "json_object"}`
- **Prompt style:** System prompt + few-shot examples + structured JSON output schema

**Reproducibility:** The full fine-tuning pipeline is available at:
- Training data: `data/processed/finetune_train.jsonl`
- Fine-tuning script: `src/models/train.py` → `launch_finetune()`
- Prompt templates: `src/models/llm_model.py` → `SYSTEM_PROMPT`, `FEW_SHOT_EXAMPLES`

---

## Bias, Risks and Limitations

### Known Limitations
- **Synthetic data only:** Trained exclusively on programmatically generated scenarios; real-world traffic has different distributions (weather, road geometry, human behaviour).
- **No temporal reasoning:** Each call is stateless; the model does not track vehicle trajectories over time.
- **API dependency:** Requires internet access to the OpenAI API; not suitable for fully offline edge deployment.
- **Fine-tuning opacity:** As an API-fine-tuned model, raw weights are not accessible for inspection.

### Potential Bias
- **Direction bias:** Because conflict pairs are symmetric in the training data, all 4 directions should be treated equally. Evaluated via `audit_bias()` in `src/evaluation/evaluate.py`.
- **Speed bias:** Priority is assigned purely by speed; in reality, vehicle type (e.g., emergency vehicles) should also factor in.

### Risks
- Using model outputs as direct control signals without human validation could be hazardous.
- The model may hallucinate vehicle IDs or invent conflicts in ambiguous edge cases.

---

## Environmental Impact

Energy consumption is tracked using **CodeCarbon** during training and evaluation runs.

| Item | Estimated Value |
|---|---|
| Hardware | Cloud GPU (OpenAI infrastructure) |
| CO₂ per fine-tuning run | < 0.5 kg CO₂-eq (estimated) |
| CO₂ per evaluation run | < 0.01 kg CO₂-eq |

---

## Citation

If you use this model or the associated dataset, please cite:

```bibtex
@article{masri2025llm,
  title   = {Large Language Models (LLMs) as Traffic Control Systems
             at Urban Intersections: A New Paradigm},
  author  = {Masri, Sari and Ashqar, Huthaifa I. and Elhenawy, Mohammed},
  journal = {arXiv preprint arXiv:2411.10869},
  year    = {2025},
  url     = {https://arxiv.org/abs/2411.10869}
}
```

---

## How to Use

### Zero-shot (no fine-tuning required)

```python
from src.models.llm_model import IntersectionLLM

llm = IntersectionLLM(model="gpt-4o-mini", few_shot=True)
result = llm.predict({
    "vehicles": [
        {"vehicle_id": "V001", "lane": 1, "speed": 60,
         "distance_to_intersection": 50, "direction": "north", "destination": "A"},
        {"vehicle_id": "V002", "lane": 1, "speed": 50,
         "distance_to_intersection": 55, "direction": "south", "destination": "C"},
    ]
})
print(result)
```

### Rule-based baseline (no API key needed)

```python
from src.poc.conflict_detection import analyze_intersection

decision = analyze_intersection([
    {"vehicle_id": "V001", "lane": 1, "speed": 60,
     "distance_to_intersection": 50, "direction": "north", "destination": "A"},
    {"vehicle_id": "V002", "lane": 1, "speed": 50,
     "distance_to_intersection": 55, "direction": "south", "destination": "C"},
])
print(decision.to_dict())
```

---

## Model Card Authors

CSC5382 – AI for Digital Transformation Project Team

Inspired by the [Hugging Face Model Card](https://huggingface.co/docs/hub/model-cards) format and the original baseline by Masri et al. available at [sarimasri3/Intersection-Conflict-Detection](https://github.com/sarimasri3/Intersection-Conflict-Detection).
