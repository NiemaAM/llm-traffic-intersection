"""
deploy_hf.py
------------
Deploy the Streamlit app to a Hugging Face Space (Docker SDK).
Uploads all files needed for standalone operation.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE = os.environ["HF_SPACE"]

api = HfApi(token=HF_TOKEN)

def upload(local: str, remote: str) -> None:
    if not Path(local).exists():
        print(f"  ⚠️  Skipping {local} (not found)")
        return
    print(f"  📤 {local} → {remote}")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=HF_SPACE,
        repo_type="space",
        commit_message=f"Deploy: {remote}",
    )

print(f"🚀 Deploying to https://huggingface.co/spaces/{HF_SPACE}")

# ── Dockerfile for HF Space ───────────────────────────────────────────────────
dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/

ENV PYTHONPATH=/app
ENV MODEL_NAME=gpt-4o-mini
ENV FEW_SHOT=true

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""
with open("/tmp/hf_Dockerfile", "w") as f:
    f.write(dockerfile)
upload("/tmp/hf_Dockerfile", "Dockerfile")

# ── Main app ──────────────────────────────────────────────────────────────────
upload("src/api/streamlit_app.py", "app.py")

# ── Requirements ─────────────────────────────────────────────────────────────
hf_requirements = """streamlit>=1.35.0
openai>=1.30.0
pandas>=2.1.0
plotly>=5.20.0
requests>=2.31.0
python-dotenv>=1.0.0
scikit-learn>=1.4.0
"""
with open("/tmp/hf_requirements.txt", "w") as f:
    f.write(hf_requirements)
upload("/tmp/hf_requirements.txt", "requirements.txt")

# ── Source modules needed at runtime ─────────────────────────────────────────
src_files = [
    # Models — uploaded to both src/models/ AND models/ for import compatibility
    ("src/models/llm_model.py",            "src/models/llm_model.py"),
    ("src/models/llm_model.py",            "models/llm_model.py"),
    ("src/models/__init__.py",             "src/models/__init__.py"),
    ("src/models/__init__.py",             "models/__init__.py"),
    # Visualization — same folder as app.py for guaranteed import
    ("src/api/conflict_detection_orig.py", "conflict_detection_orig.py"),
    ("src/api/visualization_orig.py",      "visualization_orig.py"),
    # Package inits
    ("src/__init__.py",                    "src/__init__.py"),
]

for local, remote in src_files:
    upload(local, remote)

# ── __init__.py files (create if missing) ─────────────────────────────────────
for init_path in ["src/models/__init__.py", "src/poc/__init__.py", "src/__init__.py"]:
    if not Path(init_path).exists():
        with open("/tmp/empty_init.py", "w") as f:
            f.write("")
        upload("/tmp/empty_init.py", init_path)

# ── HF Space README ───────────────────────────────────────────────────────────
readme = """---
title: LLM Traffic Intersection Conflict Resolver
emoji: 🚦
colorFrom: red
colorTo: blue
sdk: docker
pinned: true
---

# 🚦 LLM-Driven Intersection Conflict Resolver

Production serving of **GPT-4o-mini (fine-tuned)** for traffic intersection conflict detection.

- **Model:** Fine-tuned GPT-4o-mini (Accuracy: 78.3% | F1: 0.78)
- **Baseline:** Masri et al. (2025) — [arXiv:2411.10869](https://arxiv.org/abs/2411.10869)
- **Course:** CSC5382 – AI for Digital Transformation

## Setup
Add these secrets in Space Settings:
- `OPENAI_API_KEY` — your OpenAI API key
- `FINE_TUNED_MODEL_ID` — `ft:gpt-4o-mini-2024-07-18:personal::DWL89pFu`
- `MODEL_NAME` — `gpt-4o-mini`
"""
with open("/tmp/README.md", "w") as f:
    f.write(readme)
upload("/tmp/README.md", "README.md")

print(f"\n✅ Deployed → https://huggingface.co/spaces/{HF_SPACE}")
print("   Make sure these Space secrets are set:")
print("   - OPENAI_API_KEY")
print("   - FINE_TUNED_MODEL_ID = ft:gpt-4o-mini-2024-07-18:personal::DWL89pFu")
print("   - MODEL_NAME = gpt-4o-mini")