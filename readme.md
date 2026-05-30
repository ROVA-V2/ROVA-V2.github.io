# RoVA-V2 — Robust-TO

Reference implementation of **Robust-TO** (*Robust Video Understanding with Tool
Orchestration*), the agentic framework from *“Confidence-Aware Tool Orchestration
for Robust Video Understanding.”*

---

## Method overview

Robust-TO addresses the **Blind Trust Problem**: video-QA pipelines delegate to
visual tools without knowing how reliable those tools are on degraded input. Its
unifying principle:

> **Every visual tool reports not only what it sees, but how much it trusts what it sees.**

The pipeline is a **single feed-forward pass** (Fig. 2 in the paper) — there are
**no inference-time loops**. When the video is clean, every fact lands in the
high-reliability tier and the pipeline reduces to ordinary multi-fact reasoning
with no overhead (Sec. 3.1).

```
Video V~ + Query q
        │
        ▼
Disturbance-Aware Adaptive Perception            (Section 3.2)
   assess_quality   → per-frame disturbance d(fi)            (Eq. 2)
   select_frames    → top-K trustworthy frames + pool P      (Eq. 3)
   decompose q      → atomic sub-queries {sq1..sqm}          (Text+Frame, App. E.1)
   route each sqi   → tool by semantic type + dominant corruption (Table 18, App. E.2)
   call tool        → (result, confidence) per sub-query      (Eq. 4)
        │  tagged evidence set  F = {(r_i, c_i, src_i)}
        ▼
Confidence-Weighted Evidence Synthesis           (Section 3.1)
   single reasoning pass over three reliability tiers (HIGH / MEDIUM / LOW)
        │
        ▼
   Answer a   (+ GRPO confidence-cost reward during training, Section 3.3)
```

### Unified (result, confidence) interface — Eq. 4

Every tool returns a result tied to its source frames plus a calibrated
confidence that multiplies the tool's own certainty by the **input reliability**
of those frames:

```
(r_j, c_j) = T_j(F, sq)
c_j = c_intrinsic_j  ×  ρ(F)
ρ(F) = worst-K mean of (1 − d(f)),   K = ⌈|F| / 3⌉
```

`ρ(F)` is the mean of the **K = ⌈n/3⌉ smallest** values of `(1 − d(f))` (the most
corrupted frames), so a single clean frame cannot mask catastrophic corruption in
the others (Eq. 4; ablated in Table 11 — uniform mean loses 3.7 pts on clean
UrbanVideo). Intrinsic confidences are clipped to `[0.01, 1.0]`.

### Three-tier synthesis (App. E.3)

```
HIGH:   c_j ≥ 0.7  AND  d < 0.3
LOW:    c_j < 0.3  OR   d ≥ 0.7
MEDIUM: otherwise
```

The answer is built primarily from HIGH-tier facts; MEDIUM facts are used only if
consistent with HIGH; LOW facts only when no HIGH evidence exists (with explicit
uncertainty). Output format: `<think>...</think><answer>X</answer>`.

---

## Tool library (Table 17 — reference only)

The table below is the paper's **example** tool library (3 selection + 5
perception tools) — **for reference only**. Per Table 17, *“users may define
their own tool sets and assign arbitrary costs, as long as every tool conforms
to the unified `(result, confidence)` interface.”* The whole visual tool library
is therefore **user-definable**: swap tools in/out, rename them, add new ones,
and set your own costs. The two-stage router depends only on each tool's declared
semantic type + disturbance profile (Sec. 3.2), so the library is fully
pluggable.

**What this release ships is one runnable default**, not a fixed library: the
three perception-side components needed to run end-to-end — `det`
(`detect_objects`), `ocr` (`read_text`), and the **APE detection service**
(`ape_tools/`) that backs detection. Everything else is a plug-and-play template.

| Tool | Category | Cost | Status in this release |
|---|---|---|---|
| `assess_quality`   | Selection  | 0.10 | shipped (Eq. 2) |
| `select_frames`    | Selection  | 0.15 | shipped (Eq. 3) |
| `retrieve_frames`  | Selection  | 0.20 | shipped (pool-P retrieval; case study Tab. 7) |
| **`detect_objects`** | Perception | 0.50 | **shipped — APE backend (det)** |
| **`read_text`**      | Perception | 0.25 | **shipped — OCR via host VLM (ocr)** |
| `caption_frame`    | Perception | 0.30 | plug-and-play template (add your own) |
| `track_temporal`   | Perception | 0.70 | plug-and-play template (add your own) |
| `recognize_action` | Perception | 0.60 | plug-and-play template (add your own) |

Costs are **reference values** (calibrated against `caption_frame` as the unit);
your own tools may declare their own. Unregistered tools fall back to cost 0.5 in
the reward.

### Define your own tool library (plug-and-play, arbitrary costs)

```python
from rovid_pipeline.tools.perception_tools import PluggableTool, build_perception_tools

class MyCaptioner(PluggableTool):
    name = "caption_frame"
    cost = 0.30                                    # arbitrary, your choice (Table 17)
    def _run(self, frames, sub_query, disturbance_scores):
        caption = my_model(frames, sub_query)      # your backend
        return caption, my_confidence_in_[0,1]     # (result, c_intrinsic)

# Register custom tools (and/or arbitrary costs) — nothing else changes:
tools = build_perception_tools(
    vlm_fn,
    extra_tools={"caption_frame": MyCaptioner()},
    extra_costs={"my_depth_tool": 0.45},           # or set per-tool `cost`
)
# ... or pass extra_tools=/extra_costs= straight to RobustTOPipeline(...)
```

The router, the Eq. 4 confidence coupling, and the GRPO confidence-cost reward
all work unchanged once a tool is registered — the orchestration layer never
needs editing.

> **Detection backbone note.** Paper Section B.1 names **GroundingDINO-T** for
> `detect_objects`; this release wraps **APE** instead (the detector actually
> used in the released code). The intrinsic confidence still follows B.1 — the
> mean detection score over returned boxes, discarding boxes below 0.3. To
> reproduce with GroundingDINO-T, swap the backend in `DetectObjects._run`; the
> `(label, box, score)` contract is all the tool depends on.


## GRPO confidence-cost reward (Section 3.3)

```
R_cc(c_j, T_j) = c_j − λ·cost(T_j)                         # per call   (Eq. 5),  λ = 0.5
R_cc^total(τ)  = (1/N_call) Σ_k R_cc                        # trajectory (Eq. 6)
R_min-sq       = exp(−α·max(0, n − N*))                     # (Eq. 7),    α = 0.2
R_qual         = (1 − exp(−β·n/max(N*,1)))·mean_conf        # (Eq. 8),    β = 1.0
R_subq         = ½(R_min-sq + R_qual)                       # (Eq. 9)
R_total        = R_acc + w·(R_subq + R_cc^total + R_fmt)    # (Eq. 10),   w = 1/3
```

`R_acc ∈ {−1, +1}`, `R_fmt ∈ {0, 1}`. The shared `w = 1/3` keeps the auxiliary
sum's magnitude ≤ 1, matching `|R_acc|` (Eq. 10 / Sec. 3.3). Table 19 lists
`w_subq = w_cc = w_fmt = 0.3`, which agrees with `w = 1/3` up to rounding; pass
`w_subq/w_cc/w_fmt=0.3` to `compute_trajectory_reward` to reproduce Table 19
verbatim. `N*(q)` is estimated once per question by a **frozen** text-only VLM
(Qwen2.5-7B-Instruct in the paper) to prevent reward gaming — pass it via
`estimator_fn`, distinct from the policy VLM.

---

## Installation

```bash
# 1. Host VLM (LLaVA-NeXT used here; swap for your own VLM)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
conda create -n robustto python=3.10 -y && conda activate robustto
pip install --upgrade pip
pip install -e ".[train]"
pip install faiss-cpu networkx torch==2.1.2 torchaudio decord

# 2. APE detection backend (powers detect_objects)
git clone https://github.com/shenyunhang/APE
cd APE && pip3 install -r requirements.txt && python3 -m pip install -e .

# 3. Copy this package + the APE service into place
cp -r rovid_pipeline/ <LLaVA-NeXT-root>/
cp -r ape_tools/      <APE-root>/demo/

# 4. Start the APE service (TCP port 9999)
cd <APE-root> && python demo/ape_service.py
```

---

## Usage

```bash
python -m rovid_pipeline.rovid_pipeline \
    --video    /path/to/video.mp4 \
    --question "In what order does the drone pass the landmarks?" \
    --answer   "B"          # optional, enables GRPO reward computation
```

```python
from rovid_pipeline import RobustTOPipeline, process_video

frames, _, _ = process_video("video.mp4", max_frames=32, force_sample=True)

pipeline = RobustTOPipeline(agent_fn=agent_fn, vlm_fn=vlm_inference_fn)
result   = pipeline.run(frames, query="What is happening?", ground_truth="A")

print(result["answer"])            # e.g. "A"
print(result["reasoning"])         # <think> content
print(result["reward"].R_total)    # composite GRPO reward (Eq. 10)
```

`RoVidPipeline` is kept as a backward-compatible alias for `RobustTOPipeline`.

### Adapting to another VLM

In `rovid_pipeline/rovid_pipeline.py`, replace the blocks marked
`# [REPLACE MODEL]` and `# [REPLACE INFERENCE]`, and supply:

```python
def agent_fn(prompt: str) -> str: ...                       # text-only orchestration
def vlm_inference_fn(prompt: str, frames) -> str: ...       # vision + language
```

---

## Repository structure

```
rovid_pipeline/
├── rovid_pipeline.py        # RobustTOPipeline (alias RoVidPipeline) + CLI
├── reward.py                # GRPO confidence-cost reward (Eqs. 5–10)
├── tools/
│   ├── base.py              # unified (result, confidence) interface (Eq. 4, worst-K ρ)
│   ├── selection_tools.py   # assess_quality (Eq. 2), select_frames (Eq. 3), retrieve_frames
│   └── perception_tools.py  # detect_objects (APE), read_text (OCR) + plug-and-play templates
└── stages/
    ├── perception.py        # Disturbance-Aware Adaptive Perception (Section 3.2)
    └── synthesis.py         # Confidence-Weighted Evidence Synthesis (Section 3.1)

ape_tools/
├── ape_api.py               # APE detection inference (returns label, box, score)
└── ape_service.py           # socket server for APE (TCP 9999, length-prefixed framing)

evals/
├── generate_urbanvideo.py   # UrbanVideo-Bench (LP/CF/PE/AG), MCQ accuracy
└── generate_vsi.py          # VSI-Bench (RDist/RDir/RP/AO), MCQ accuracy
```

---

## Evaluation

The paper evaluates on **UrbanVideo-Bench** and **VSI-Bench** (clean and
RoVA-V1-corrupted variants), reporting multiple-choice accuracy (Section B.3).

```bash
# UrbanVideo-Bench (clean)
python evals/generate_urbanvideo.py --data_path /path/to/UrbanVideo-Bench

# UrbanVideo-Bench under a RoVA-V1 corruption (MB / GN / GL / Occ / LL)
python evals/generate_urbanvideo.py --data_path /path/to/UrbanVideo-Bench --corruption GL

# VSI-Bench
python evals/generate_vsi.py --data_path /path/to/VSI-Bench
```

Each script prints per-task and average accuracy and writes predictions +
summary JSON to `--output_dir`. The dataset loaders assume a simple MCQ schema
(see each script's docstring); adapt `load_records()` to your local release.

---

## Key hyperparameters (Table 16)

| Group | Parameter | Value |
|---|---|---|
| Disturbance (Eqs. 11–13) | τ_blur / μ_ref / τ_edge | 500 / 0.5 / 30 |
| Disturbance (Eq. 2) | channel weights | equal (1:1:1) after min-max norm |
| Selection (Eq. 3) | θ_rel / θ_sim / K | 0.55 / 0.30 / K∈[4,12] |
| Confidence (Eq. 4) | ρ(F) aggregator / clip | worst-K, K=⌈n/3⌉ / [0.01, 1.0] |
| Synthesis (App. E.3) | HIGH / LOW tiers | c≥0.7 ∧ d<0.3 / c<0.3 ∨ d≥0.7 |
| Reward (Eqs. 5–10) | λ / α / β / w | 0.5 / 0.2 / 1.0 / 1/3 |
| Reward | N* estimator | frozen Qwen2.5-7B-Instruct (text-only) |
```
