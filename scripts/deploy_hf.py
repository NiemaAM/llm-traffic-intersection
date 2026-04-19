"""
deploy_hf.py
------------
Deploy the Streamlit app to a Hugging Face Space.
Uploads: app.py, requirements.txt, README.md, src/ modules needed at runtime.
Called from CI/CD pipeline on merge to main.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE = os.environ["HF_SPACE"]   # e.g. "NiemaAM/traffic-intersection"

api = HfApi(token=HF_TOKEN)

def upload(local: str, remote: str, msg: str) -> None:
    print(f"  📤 {local} → {remote}")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=HF_SPACE,
        repo_type="space",
        commit_message=msg,
    )

print(f"🚀 Deploying to https://huggingface.co/spaces/{HF_SPACE}")

# ── Main app ──────────────────────────────────────────────────────────────────
upload("src/api/streamlit_app.py", "app.py", "Deploy Streamlit app")

# ── Requirements (HF-friendly subset — no heavy MLOps deps) ──────────────────
hf_requirements = """streamlit>=1.35.0
openai>=1.30.0
pandas>=2.1.0
plotly>=5.20.0
requests>=2.31.0
python-dotenv>=1.0.0
"""
with open("/tmp/hf_requirements.txt", "w") as f:
    f.write(hf_requirements)
upload("/tmp/hf_requirements.txt", "requirements.txt", "Update requirements")

# ── Source modules needed at runtime ──────────────────────────────────────────
src_files = [
    ("src/poc/conflict_detection_orig.py", "src/poc/conflict_detection_orig.py"),
    ("src/poc/visualization_orig.py",      "src/poc/visualization_orig.py"),
]
for local, remote in src_files:
    if Path(local).exists():
        upload(local, remote, "Update source modules")

# ── HF Space README (sets SDK and env vars) ───────────────────────────────────
readme = f"""---
title: LLM Traffic Intersection Conflict Resolver
emoji: 🚦
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
---

# 🚦 LLM-Driven Intersection Conflict Resolver

Production serving of **GPT-4o-mini (fine-tuned)** for traffic intersection conflict detection.

- **Model:** `ft:gpt-4o-mini-2024-07-18` (fine-tuned, F1=0.78)
- **Baseline:** Masri et al. (2025) — [arXiv:2411.10869](https://arxiv.org/abs/2411.10869)
- **Course:** CSC5382 – AI for Digital Transformation

## Usage
Select a preset scenario or build your own, then click **Analyze Intersection**.

> ⚠️ Requires `OPENAI_API_KEY` and `FINE_TUNED_MODEL_ID` set as Space secrets.
"""
with open("/tmp/README.md", "w") as f:
    f.write(readme)
upload("/tmp/README.md", "README.md", "Update Space README")

print(f"\n✅ Deployed → https://huggingface.co/spaces/{HF_SPACE}")
print("   Set these secrets in Space settings:")
print("   - OPENAI_API_KEY")
print("   - FINE_TUNED_MODEL_ID")
print("   - MODEL_NAME=gpt-4o-mini")
print("   - API_URL=  (leave empty — HF Space runs standalone)")
