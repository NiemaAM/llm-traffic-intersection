"""
Generate architecture diagram for LLM Traffic Intersection project.
Style: AWS-style zones with numbered badges, dashed borders, service boxes.
Output: reports/figures/architecture.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "figures" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PINK = "#C2185B"
TEAL = "#00695C"
BLUE = "#1565C0"
GREEN = "#2E7D32"
GRAY = "#546E7A"
PURPLE = "#6A1B9A"
AMBER = "#E65100"
DBLUE = "#0D47A1"  # dark blue for CI/CD

# Background tints (for service boxes)
T_PINK = "#FCE4EC"
T_TEAL = "#E0F2F1"
T_BLUE = "#E3F2FD"
T_GREEN = "#E8F5E9"
T_GRAY = "#ECEFF1"
T_PURPLE = "#EDE7F6"
T_AMBER = "#FFF3E0"
T_DBLUE = "#E8EAF6"

fig, ax = plt.subplots(figsize=(24, 18))
ax.set_xlim(0, 24)
ax.set_ylim(0, 18)
ax.axis("off")
fig.patch.set_facecolor("white")


# ── Helpers ───────────────────────────────────────────────────────────────────


def badge(x, y, num, color=BLUE):
    """Numbered blue square badge (AWS-style)."""
    b = FancyBboxPatch(
        (x, y),
        0.62,
        0.62,
        boxstyle="square,pad=0",
        facecolor=color,
        edgecolor="white",
        lw=1.8,
        zorder=15,
    )
    ax.add_patch(b)
    ax.text(
        x + 0.31,
        y + 0.31,
        str(num),
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        zorder=16,
    )


def zone(x, y, w, h, label=None, color=GRAY, lw=1.6):
    """Dashed zone rectangle with optional italic label."""
    r = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor=color, lw=lw, linestyle="--", zorder=2
    )
    ax.add_patch(r)
    if label:
        ax.text(
            x + 0.22,
            y + h - 0.12,
            label,
            fontsize=9,
            style="italic",
            fontweight="bold",
            color=color,
            va="top",
            zorder=5,
        )


def svc(x, y, w, h, label, sub="", bg="white", border=GRAY):
    """Service box with bold label and optional italic subtitle."""
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.07", facecolor=bg, edgecolor=border, lw=1.6, zorder=4
    )
    ax.add_patch(p)
    if sub:
        ax.text(
            x + w / 2,
            y + h / 2 + 0.14,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=border,
            zorder=5,
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 0.15,
            sub,
            ha="center",
            va="center",
            fontsize=6.3,
            style="italic",
            color=GRAY,
            zorder=5,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=border,
            zorder=5,
        )


def arr(x1, y1, x2, y2, label="", color=GRAY, rad=0.0, lw=1.5):
    """Arrow with optional label on a white background chip."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=6,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + 0.07,
            label,
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(
    12,
    17.75,
    "LLM Traffic Intersection — System Architecture",
    ha="center",
    va="top",
    fontsize=17,
    fontweight="bold",
    color="#212121",
)
ax.text(
    12,
    17.2,
    "Milestones 1 to 6  ·  ZenML · OpenAI · MLflow · FastAPI · Prometheus · GitHub Actions",
    ha="center",
    va="top",
    fontsize=9.5,
    color=GRAY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TOP ROW — 3 setup boxes + ML Platform zone
# Mirrors: CloudFormation | Service Catalog zone | CloudFormation
# ═══════════════════════════════════════════════════════════════════════════════

# [1] Project configuration (left standalone box)
svc(
    0.3,
    15.0,
    3.8,
    1.75,
    "Project Configuration",
    "pyproject.toml · .env · ruff · black\nZenML Stack · requirements.txt",
    bg=T_PINK,
    border=PINK,
)
badge(0.3, 16.5, 1, BLUE)

# Zone: ML Platform (center dashed zone)
zone(4.5, 14.5, 14.5, 2.75, "ZenML ML Platform", TEAL, lw=2.0)
badge(4.5, 17.0, 2, BLUE)
svc(
    5.0,
    15.0,
    13.5,
    1.75,
    "ZenML Pipeline Definitions",
    "data_pipeline · training_pipeline · evaluation_pipeline · serving_pipeline · monitoring_pipeline",
    bg=T_TEAL,
    border=TEAL,
)

# [3] MLflow (right standalone box)
svc(
    19.5,
    15.0,
    4.2,
    1.75,
    "MLflow Tracking & Registry",
    "Experiments · Params · Metrics\nModel Registry · Artifacts",
    bg=T_PINK,
    border=PINK,
)
badge(19.5, 16.5, 3, BLUE)

# Arrows between top boxes and the zone
arr(4.1, 15.88, 4.5, 15.88, "Configure stack", PINK)
arr(19.5, 15.88, 19.0, 15.88, "Register runs", PINK)

# Deploy arrow: ML Platform zone → outer project zone
arr(11.75, 14.5, 11.75, 13.9, "Deploy", TEAL, lw=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTER ZONE — ZenML Project  [4]
# Mirrors: "Amazon SageMaker project"
# ═══════════════════════════════════════════════════════════════════════════════
zone(0.3, 2.9, 23.4, 11.0, "ZenML Orchestrated Project", TEAL, lw=2.2)
badge(0.3, 13.45, 4, BLUE)


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT INNER ZONE — Data, Training & Serving Workflow  [5]
# Mirrors: "Data transformation and ingestion workflow"
# ═══════════════════════════════════════════════════════════════════════════════
zone(0.6, 3.2, 17.8, 10.4, "Data ingestion, training and serving workflow", GRAY, lw=1.6)
badge(0.6, 13.15, 5, BLUE)

# ── Row 1: Data pipeline ──────────────────────────────────────────────────────
svc(
    1.0, 11.5, 2.8, 1.4, "Raw Data", "generate_data.py\nseed 42 / 123 / 999", bg=T_BLUE, border=BLUE
)
arr(3.8, 12.2, 4.3, 12.2, "Upload", GRAY)

svc(
    4.3,
    11.5,
    2.8,
    1.4,
    "Data Validation",
    "validate_data.py\nGreat Expectations",
    bg=T_BLUE,
    border=BLUE,
)
arr(7.1, 12.2, 7.6, 12.2, "Validate", GRAY)

svc(
    7.6,
    11.5,
    2.8,
    1.4,
    "Feature Engineering",
    "preprocess.py\n6 sklearn transformers",
    bg=T_BLUE,
    border=BLUE,
)
arr(10.4, 12.2, 10.9, 12.2, "Process", GRAY)

svc(
    10.9, 11.5, 2.8, 1.4, "DVC Versioning", "raw/ · processed/\n.dvc hashes", bg=T_BLUE, border=BLUE
)
arr(13.7, 12.2, 14.2, 12.2, "Ingest data", GREEN)

svc(
    14.2,
    11.5,
    3.6,
    1.4,
    "Feast Feature Store",
    "SQLite online store\nfeatures.parquet",
    bg=T_GREEN,
    border=GREEN,
)

# ── Row 2: Training pipeline ──────────────────────────────────────────────────
svc(
    1.0,
    9.6,
    2.8,
    1.4,
    "Masri JSONL",
    "train_data.jsonl\n~800 scenarios",
    bg=T_PURPLE,
    border=PURPLE,
)
arr(3.8, 10.3, 4.3, 10.3, "Format", PURPLE)

svc(
    4.3,
    9.6,
    2.8,
    1.4,
    "OpenAI Fine-Tune",
    "gpt-4o-mini → FT API\nft:DX7kzKtB",
    bg=T_PINK,
    border=PINK,
)
arr(7.1, 10.3, 7.6, 10.3, "FT Job", PINK)

svc(7.6, 9.6, 2.8, 1.4, "Fine-Tuned Model", "ft:gpt-4o-mini\nDX7kzKtB", bg=T_PINK, border=PINK)
arr(10.4, 10.3, 10.9, 10.3, "Evaluate", PURPLE)

svc(
    10.9,
    9.6,
    2.8,
    1.4,
    "Offline Evaluation",
    "9 combos · seed=999\nAcc · F1 · FNR",
    bg=T_PURPLE,
    border=PURPLE,
)
arr(13.7, 10.3, 14.2, 10.3, "Log", AMBER)

svc(
    14.2,
    9.6,
    3.6,
    1.4,
    "MLflow Experiments",
    "Metrics · Artifacts\nCodeCarbon emissions",
    bg=T_AMBER,
    border=AMBER,
)

# Vertical: Data row → Training row
arr(2.4, 11.5, 2.4, 11.0, "", GRAY)

# ── Row 3: Production serving ─────────────────────────────────────────────────
svc(
    1.0,
    7.7,
    2.8,
    1.4,
    "Rule Engine",
    "conflict_detection.py\nGround truth + fallback",
    bg=T_GREEN,
    border=GREEN,
)
arr(3.8, 8.4, 4.3, 8.4, "Fallback", GREEN)

svc(
    4.3,
    7.7,
    2.8,
    1.4,
    "IntersectionLLM",
    "llm_model.py\nunified predict()",
    bg=T_AMBER,
    border=AMBER,
)
arr(7.1, 8.4, 7.6, 8.4, "Serve", AMBER)

svc(7.6, 7.7, 2.8, 1.4, "FastAPI :8000", "app.py\n/predict · /health", bg=T_AMBER, border=AMBER)
arr(10.4, 8.4, 10.9, 8.4, "Web UI", AMBER)

svc(
    10.9,
    7.7,
    2.8,
    1.4,
    "Streamlit :8501",
    "streamlit_app.py\ninteractive dashboard",
    bg=T_AMBER,
    border=AMBER,
)
arr(13.7, 8.4, 14.2, 8.4, "Deploy", BLUE)

svc(
    14.2,
    7.7,
    3.6,
    1.4,
    "HuggingFace Spaces",
    "NiemaAM/traffic\nllm-traffic.streamlit.app",
    bg=T_BLUE,
    border=BLUE,
)

# Training → Serving: model load at startup
arr(8.4, 9.6, 5.7, 9.0, "Load model", PINK, rad=-0.12)

# ── Row 4: Prediction logging ─────────────────────────────────────────────────
svc(
    1.0,
    5.8,
    2.8,
    1.4,
    "Prediction Logger",
    "log_prediction_event()\nJSONL append",
    bg=T_GRAY,
    border=GRAY,
)
arr(3.8, 6.5, 4.3, 6.5, "", GRAY)

svc(
    4.3,
    5.8,
    2.8,
    1.4,
    "predictions.jsonl",
    "reports/monitoring/\npersistent log",
    bg=T_GRAY,
    border=GRAY,
)
arr(7.1, 6.5, 7.6, 6.5, "", GRAY)

svc(
    7.6,
    5.8,
    2.8,
    1.4,
    "Ground Truth",
    "_rule_label()\nrule engine inline",
    bg=T_GREEN,
    border=GREEN,
)
arr(10.4, 6.5, 10.9, 6.5, "", GRAY)

svc(
    10.9,
    5.8,
    2.8,
    1.4,
    "Drift Detection",
    "Evidently + KS-test\ndrift_report.html",
    bg=T_GRAY,
    border=GRAY,
)
arr(13.7, 6.5, 14.2, 6.5, "", PURPLE)

svc(
    14.2,
    5.8,
    3.6,
    1.4,
    "CL Trigger",
    "FNR · Recall · Drift\nwindow = 100 preds",
    bg=T_PURPLE,
    border=PURPLE,
)

# FastAPI → Logger
arr(9.0, 7.7, 2.4, 7.2, "Log prediction", GRAY, rad=0.18)


# ── Row 5: Monitoring metrics ─────────────────────────────────────────────────
svc(
    1.0,
    3.9,
    2.8,
    1.4,
    "Prometheus :9090",
    "8 metric gauges\nFNR · Recall · Latency",
    bg=T_PURPLE,
    border=PURPLE,
)
arr(3.8, 4.6, 4.3, 4.6, "Scrape", PURPLE)

svc(
    4.3,
    3.9,
    2.8,
    1.4,
    "Grafana :3001",
    "Dashboards\nConflict rate · FNR",
    bg=T_PURPLE,
    border=PURPLE,
)
arr(7.1, 4.6, 7.6, 4.6, "", PURPLE)

svc(
    7.6,
    3.9,
    2.8,
    1.4,
    "Arize Phoenix",
    "A/B · Bias · Robust\nExplainability",
    bg=T_AMBER,
    border=AMBER,
)
arr(10.4, 4.6, 10.9, 4.6, "", PURPLE)

svc(
    10.9,
    3.9,
    2.8,
    1.4,
    "ZenML Monitoring",
    "monitoring_pipeline.py\n7 steps",
    bg=T_PURPLE,
    border=PURPLE,
)
arr(13.7, 4.6, 14.2, 4.6, "", PURPLE)

svc(
    14.2,
    3.9,
    3.6,
    1.4,
    "Retrain Signal",
    "trigger_log.jsonl\n→ training_pipeline",
    bg=T_PURPLE,
    border=PURPLE,
)

# Logger → Prometheus
arr(2.4, 5.8, 2.4, 5.3, "Feed", PURPLE)

# CL Trigger → Monitoring pipeline
arr(15.0, 5.8, 13.25, 5.3, "Evaluate", PURPLE, rad=0.12)

# Retrain signal → Training pipeline (feedback loop)
arr(16.0, 5.8, 16.0, 10.3, "retrain →", PURPLE, rad=0.0, lw=1.8)
arr(16.0, 10.3, 14.5, 10.3, "", PURPLE, lw=1.8)


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT INNER ZONE — CI/CD Automation  [7]
# Mirrors: "CI/CD automation" zone in the AWS diagram
# ═══════════════════════════════════════════════════════════════════════════════
zone(18.7, 3.2, 5.2, 10.4, "CI/CD automation", DBLUE, lw=1.6)
badge(18.7, 13.15, 7, BLUE)

svc(
    19.0,
    11.5,
    4.6,
    1.4,
    "GitHub Repository",
    "Code versioning\npyproject.toml · ruff · black",
    bg=T_DBLUE,
    border=DBLUE,
)
arr(21.3, 11.5, 21.3, 11.0, "", DBLUE)

svc(
    19.0,
    9.6,
    4.6,
    1.4,
    "GitHub Actions",
    "lint → test → data-pipeline\nbuild → deploy workflow",
    bg=T_DBLUE,
    border=DBLUE,
)
arr(21.3, 9.6, 21.3, 9.1, "", DBLUE)

svc(
    19.0,
    7.7,
    4.6,
    1.4,
    "Docker Build",
    "multi-stage Dockerfile\nGHCR push",
    bg=T_DBLUE,
    border=DBLUE,
)
arr(21.3, 7.7, 21.3, 7.2, "", DBLUE)

svc(
    19.0,
    5.8,
    4.6,
    1.4,
    "HuggingFace Deploy",
    "deploy_hf.py\nSpaces: NiemaAM/traffic",
    bg=T_DBLUE,
    border=DBLUE,
)
arr(21.3, 5.8, 21.3, 5.3, "", DBLUE)

svc(
    19.0,
    3.9,
    4.6,
    1.4,
    "Streamlit Cloud",
    "llm-traffic-intersection\n.streamlit.app",
    bg=T_DBLUE,
    border=DBLUE,
)

# Build arrow: CI/CD → Training pipeline (like the "Build" arrow in the AWS diagram)
arr(18.7, 10.3, 18.4, 10.3, "Build", DBLUE)


# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM STANDALONE BOX — Online Evaluation summary
# ═══════════════════════════════════════════════════════════════════════════════
zone(
    0.3,
    0.2,
    23.4,
    2.5,
    "Online Evaluation  (Arize Phoenix + reports/online_evaluation/)",
    AMBER,
    lw=1.8,
)
badge(0.3, 2.25, 6, "#E65100")

bw, bh, by, gap = 3.6, 1.5, 0.45, 0.28
for i, (lbl, sub, border, bg) in enumerate(
    [
        ("A/B Testing", "few-shot vs fine-tuned\nchi²=4.64  p=0.031", AMBER, T_AMBER),
        ("Bias Audit", "7 dimensions\nFNR-disp=0.000", AMBER, T_AMBER),
        ("Robustness Tests", "5 features · flip-rate\nstability = 96.3%", AMBER, T_AMBER),
        ("Explainability", "LIME · feature rank\nrule agreement = 56.7%", AMBER, T_AMBER),
        ("Drift Detection", "Evidently + KS-test\ndrift_report.html", GRAY, T_GRAY),
        ("CL Decision", "FNR>0.08 · Recall<0.92\nretrain signal", PURPLE, T_PURPLE),
    ]
):
    svc(0.5 + i * (bw + gap), by, bw, bh, lbl, sub, bg=bg, border=border)

# predictions.jsonl feed → Online Evaluation zone
arr(5.8, 3.9, 5.8, 2.7, "JSONL feed", PURPLE)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.15)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}")
