# RoVA-V2: RoVid Pipeline

> **Submitted to NeurIPS 2026 — Do not distribute.**

Implementation of **RoVid** — the two-stage video reasoning framework described in the paper.

This release ships the complete tool library (3 selection tools + 5 perception tools, matching Table 17) that exercises every paper component: the unified confidence interface (Eq. 1), disturbance-aware selection (Eqs. 2–3), confidence-driven re-invocation (Eq. 4, Loop 1), uncertainty-triggered look-back (Eq. 5, Loop 3), and the GRPO confidence-cost reward (Eqs. 6–11). Additional perception tools can be added without touching the orchestration layer — see `tools/perception_tools.py`.

---

## Method Overview

RoVid addresses the *Blind Trust Problem*: standard video-QA pipelines delegate to visual tools without knowing how reliable those tools are on degraded input. RoVid's unifying principle:

> **Every visual tool reports not only what it sees, but how much it trusts what it sees.**

### Two Stages, Four Steps, Three Loops

```
Video + Query
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1 · Tool-Driven Visual Gathering  (Section 0.3)   │
│   Step 1.1  assess_quality   → d(fi) per frame  (Eq.2)  │
│   Step 1.2  select_frames    → top-K + pool P   (Eq.3)  │
│   Step 1.3  sub-query decomposition + tool dispatch     │
│   Step 1.4  Loop 1: confidence-driven re-invocation     │
│             (Eq.4)  — inference-time only               │
│   → Tagged fact set F = {(r_i, c_i, src(f))}            │
└────────────────────────────┬────────────────────────────┘
                             │ facts with confidence + pool P
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2 · Contemplate                   (Section 0.4)   │
│   Confidence-Weighted Reasoning                         │
│   Loop 2: gap-driven active inquiry                     │
│   Loop 3: uncertainty-triggered look-back  (Eq.5)       │
│   → Final answer                                        │
└─────────────────────────────────────────────────────────┘
```

Internally Stage 1 is split between two helper modules — `Scout` (Steps 1.1–1.2) and `Perceive` (Steps 1.3–1.4) — but external code interacts with Stage 1 as a single unit via `stages/stage1.py`.

### Confidence Interface (Eq. 1)

All tools share a unified interface:

```
(r_j, c_j) = T_j(F, sq)
c_j = c_intrinsic_j × (1/|F|) × Σ(1 − d(f))
```

### Tool Library (Table 1, simplified)

This implementation provides the full five-tool perception library (Table 17) and the two-stage disturbance-aware routing (Table 18). Adding more perception tools is purely additive — `build_perception_tools()` returns a name-keyed dict that the dispatcher resolves at runtime.

| Tool | Category | Cost | Notes |
|---|---|---|---|
| `assess_quality` | Selection | 0.10 | Eq. 2 |
| `select_frames` | Selection | 0.15 | Eq. 3 |
| `retrieve_frames` | Selection | 0.20 | Used in Loop 1 Strategy B and Loop 3 |
| `detect_objects` | Perception | 0.50 | APE service; discouraged under blur |
| `caption_frame` | Perception | 0.30 | VLM-based; preferred under blur |

### Training: Confidence-Cost Reward (Eq. 6–11)

```
R_cc(c_j, T_j) = c_j − λ · cost(T_j)          # per-call  (λ=0.5)
R_total_cc(τ)  = (1/N_call) Σ R_cc             # trajectory average
R_total = 1.0·R_acc + 0.2·R_subq + 0.3·R_cc + 0.1·R_fmt   # composite
```

---

## Installation

### Step 1 — LLaVA-NeXT (VLM backbone)

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
conda create -n rovid python=3.10 -y && conda activate rovid
pip install --upgrade pip
pip install -e ".[train]"
pip install faiss-cpu networkx torch==2.1.2 torchaudio decord
```

### Step 2 — APE (object detection service)

```bash
git clone https://github.com/shenyunhang/APE
cd APE && pip3 install -r requirements.txt && python3 -m pip install -e .
```

### Step 3 — Copy pipeline files

```bash
# Copy rovid_pipeline/ under LLaVA-NeXT root
cp -r rovid_pipeline/ <LLaVA-NeXT-root>/

# Copy ape_tools/ under APE/demo/
cp -r ape_tools/ <APE-root>/demo/
```

### Step 4 — Start APE service

```bash
cd <APE-root>
python demo/ape_service.py
```

---

## Usage

```bash
python -m rovid_pipeline.rovid_pipeline \
    --video  /path/to/video.mp4 \
    --question "What action does the person perform at the end?" \
    --answer  "B"          # optional, enables reward computation
```

### Python API

```python
from rovid_pipeline import RoVidPipeline, process_video

frames, _, _ = process_video("video.mp4", max_frames=32, force_sample=True)

pipeline = RoVidPipeline(agent_fn=agent_fn, vlm_fn=vlm_inference_fn)
result   = pipeline.run(frames, query="What is happening?", ground_truth="A")

print(result["answer"])          # e.g. "A"
print(result["reward"].R_total)  # composite GRPO reward
```

### Adapting to Another VLM

In `rovid_pipeline/rovid_pipeline.py`, replace the sections marked `# [REPLACE MODEL]` and `# [REPLACE INFERENCE]`:

```python
# 1. Load your model
tokenizer, model, image_processor, _ = load_your_model(...)

# 2. Implement the inference function signature:
def vlm_inference_fn(prompt: str, frames: Optional[np.ndarray]) -> str: ...
def agent_fn(prompt: str) -> str: ...
```

---

## Repository Structure

```
rovid_pipeline/
├── rovid_pipeline.py        # Main pipeline + CLI entry-point
├── reward.py                # GRPO confidence-cost reward (Eq. 6–11)
├── tools/
│   ├── base.py              # Unified (result, confidence) interface (Eq. 1)
│   ├── selection_tools.py   # assess_quality, select_frames, retrieve_frames
│   └── perception_tools.py  # 5 perception tools (Table 17)
└── stages/
    ├── scout.py             # Stage 1 helper: Steps 1.1 & 1.2
    ├── perceive.py          # Stage 1 helper: Steps 1.3 & 1.4 (Loop 1)
    ├── stage1.py            # Stage 1: Tool-Driven Visual Gathering (public API)
    └── contemplate.py       # Stage 2: Contemplate (Loops 2 & 3)

ape_tools/
├── ape_api.py               # APE detection inference
└── ape_service.py           # Socket server for APE (TCP port 9999)

evals/
├── generate_mlvu.py
├── generate_videomme.py
└── generate_longvideobench.py
```

---

## Evaluation

```bash
# VideoMME
python evals/generate_videomme.py

# MLVU
python evals/generate_mlvu.py

# LongVideoBench
python evals/generate_longvideobench.py
```
