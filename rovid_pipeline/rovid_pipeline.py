"""
Robust-TO — Main Pipeline  (Section 3)

Project   : RoVA-V2  (this release)
Method    : Robust-TO — Robust Video Understanding with Tool Orchestration
Corruption: RoVA-V1 video masker (used to generate corrupted benchmark variants)

The pipeline is a single feed-forward pass (Fig. 2), with no inference-time loops:

    Disturbance-Aware Adaptive Perception  (Section 3.2, stages/perception.py)
        assess_quality -> select_frames -> sub-query decomposition
        -> disturbance-aware tool routing -> unified (result, confidence) calls
        -> tagged evidence set  F = {(r_i, c_i, src_i)}
    Confidence-Weighted Evidence Synthesis (Section 3.1, stages/synthesis.py)
        single reasoning pass over HIGH/MEDIUM/LOW reliability tiers -> answer

When the video is clean every fact is HIGH-tier and the pipeline reduces to
ordinary multi-fact reasoning with no overhead (Section 3.1).

The host VLM is trained with GRPO using the confidence-cost reward (Section 3.3,
reward.py), computed on this single fixed-length trajectory.

Adapting to your own VLM
------------------------
Replace the blocks marked  # [REPLACE MODEL]  and  # [REPLACE INFERENCE]  and
provide two callables:
    agent_fn(prompt: str) -> str                          # text-only orchestration
    vlm_inference_fn(prompt: str, frames) -> str          # vision + language
"""

from __future__ import annotations
import argparse
import copy
import os
from typing import Callable, Dict, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

try:
    # LLaVA-Video backbone  # [REPLACE MODEL] if using a different VLM
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except Exception:
    DEFAULT_IMAGE_TOKEN = None
    IMAGE_TOKEN_INDEX = None
    conv_templates = {}
    tokenizer_image_token = None
    load_pretrained_model = None

from .stages.perception import DisturbanceAwarePerception, PerceptionOutput
from .stages.synthesis import ConfidenceWeightedSynthesis
from .tools.base import ToolBase
from .tools.perception_tools import build_perception_tools
from .tools.selection_tools import build_selection_tools
from .reward import compute_trajectory_reward, estimate_optimal_subqueries


MAX_FRAMES = 32   # max frames sampled before selection (Table 19)


# ─────────────────────────────────────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    max_frames: int = MAX_FRAMES,
    fps: int = 1,
    force_sample: bool = False,
) -> tuple[np.ndarray, str, float]:
    """Returns (frames [N,H,W,3] uint8, frame_time_str, video_duration_sec)."""
    if VideoReader is not None and cpu is not None:
        vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
        total = len(vr)
        if total == 0:
            raise RuntimeError(f"Video contains no readable frames: {video_path}")
        avg_fps = max(float(vr.get_avg_fps()), 1e-6)
        video_time = total / avg_fps
        step = max(1, round(avg_fps / fps))
        frame_idx = list(range(0, total, step))
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames or force_sample:
            sample_count = min(total, max_frames)
            frame_idx = np.linspace(0, total - 1, sample_count, dtype=int).tolist()
            frame_time = [i / avg_fps for i in frame_idx]
        time_str = ",".join(f"{t:.2f}s" for t in frame_time)
        frames = vr.get_batch(frame_idx).asnumpy()
        return frames, time_str, video_time

    if cv2 is None:
        raise RuntimeError("Neither decord nor OpenCV is installed; cannot read videos")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    avg_fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 1e-6)
    all_frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not all_frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    all_frames = np.stack(all_frames, axis=0)
    total = len(all_frames)
    video_time = total / avg_fps
    step = max(1, round(avg_fps / fps))
    frame_idx = list(range(0, total, step))
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames or force_sample:
        sample_count = min(total, max_frames)
        frame_idx = np.linspace(0, total - 1, sample_count, dtype=int).tolist()
        frame_time = [i / avg_fps for i in frame_idx]
    frames = all_frames[frame_idx]
    time_str = ",".join(f"{t:.2f}s" for t in frame_time)
    return frames, time_str, video_time


# ─────────────────────────────────────────────────────────────────────────────
# VLM backbone setup  [REPLACE MODEL / REPLACE INFERENCE]
# ─────────────────────────────────────────────────────────────────────────────

DEVICE        = "cuda"
MODEL_NAME    = "LLaVA-Video-7B-Qwen2"
CONV_TEMPLATE = "qwen_1_5"

if os.environ.get("ROVID_SKIP_MODEL_LOAD") != "1" and load_pretrained_model is not None and torch is not None:
    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            MODEL_NAME, None, "llava_qwen",
            torch_dtype="bfloat16", device_map="auto", overwrite_config={},
        )
        model.eval()
    except Exception:
        tokenizer = model = image_processor = max_length = None
else:
    tokenizer = model = image_processor = max_length = None   # test mode


def llava_inference(prompt: str, frames: Optional[np.ndarray]) -> str:
    """Call the VLM with an optional video clip.  # [REPLACE INFERENCE]"""
    if model is None or tokenizer is None or image_processor is None:
        raise RuntimeError(
            "LLaVA model is unavailable. Install the LLaVA dependency, or provide "
            "your own agent_fn/vlm_inference_fn to RobustTOPipeline."
        )
    if frames is not None and len(frames) > 0:
        question = DEFAULT_IMAGE_TOKEN + prompt
        video_tensor = image_processor.preprocess(frames, return_tensors="pt")
        video_tensor = [video_tensor["pixel_values"].to(DEVICE, torch.bfloat16)]
        modalities = ["video"]
    else:
        question = prompt
        video_tensor = None
        modalities = []
    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cont = model.generate(
            input_ids, images=video_tensor, modalities=modalities,
            do_sample=False, temperature=0, max_new_tokens=1024,
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()


def agent_fn(prompt: str) -> str:
    """Text-only LLM call for orchestration decisions."""
    return llava_inference(prompt, None)


def vlm_inference_fn(prompt: str, frames: Optional[np.ndarray]) -> str:
    """Vision+language call for perception tools."""
    return llava_inference(prompt, frames)


def build_backbone_similarity_fn() -> Callable[[np.ndarray, str], np.ndarray]:
    """
    Reuse the loaded VLM backbone for the sim() term in Eq. 3 instead of loading
    a separate CLIP model.  Falls back to neutral similarity when the model does
    not expose compatible vision/text embeddings.
    """
    def _neutral(frames: np.ndarray, query: str) -> np.ndarray:
        return np.ones(len(frames), dtype=np.float32)

    if model is None or tokenizer is None or image_processor is None:
        return _neutral

    def _encode_text(query: str):
        if not hasattr(tokenizer, "__call__"):
            return None
        tokenized = tokenizer(query, return_tensors="pt", truncation=True)
        input_ids = tokenized["input_ids"].to(next(model.parameters()).device)
        text_model = model.get_model() if hasattr(model, "get_model") else None
        embed = getattr(text_model, "embed_tokens", None) if text_model is not None else None
        if embed is None:
            embed = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed is None:
            return None
        with torch.no_grad():
            e = embed(input_ids)
        pooled = e.mean(dim=1)
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _encode_images(frames: np.ndarray):
        if not hasattr(model, "get_vision_tower"):
            return None
        try:
            vt = model.get_vision_tower()
        except Exception:
            return None
        if vt is None:
            return None
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        try:
            pv = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            pv = pv.to(device=device, dtype=dtype)
            with torch.no_grad():
                out = vt(pv)
            hidden = getattr(out, "last_hidden_state", out)
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[0]
            if hidden is None:
                return None
            pooled = hidden.mean(dim=1)
            return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        except Exception:
            return None

    def _similarity(frames: np.ndarray, query: str) -> np.ndarray:
        tf = _encode_text(query)
        imf = _encode_images(frames)
        if tf is None or imf is None:
            return _neutral(frames, query)
        tf = tf.to(imf.device, dtype=imf.dtype)
        sims = torch.matmul(imf, tf.transpose(0, 1)).squeeze(-1)
        return sims.detach().cpu().float().numpy()

    return _similarity


# ─────────────────────────────────────────────────────────────────────────────
# Robust-TO pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RobustTOPipeline:
    """
    Robust-TO  (Section 3): a single feed-forward pass over
    Disturbance-Aware Adaptive Perception (Sec 3.2) and
    Confidence-Weighted Evidence Synthesis (Sec 3.1).

    Core principle: every visual tool reports not only WHAT it sees, but HOW
    MUCH it trusts what it sees (the unified (result, confidence) interface,
    Eq. 4) — turning tool orchestration from blind delegation into reasoning
    under uncertainty.

    Parameters
    ----------
    agent_fn      : fn(prompt) -> str            (host VLM, text-only orchestration)
    vlm_fn        : fn(prompt, frames) -> str    (host VLM, vision + language)
    estimator_fn  : fn(prompt) -> str            (FROZEN VLM for N*; defaults to
                    agent_fn, but pass a separate frozen model to follow the
                    paper and avoid reward gaming — Section 3.3)
    extra_tools   : dict {name: ToolBase} of additional perception tools to
                    register (the plug-and-play extension point, Section 3.2)
    """

    def __init__(
        self,
        agent_fn,
        vlm_fn,
        ape_host: str = "0.0.0.0",
        ape_port: int = 9999,
        k_simple: int = 4,
        k_complex: int = 12,
        similarity_fn: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
        estimator_fn: Optional[Callable[[str], str]] = None,
        extra_tools: Optional[Dict[str, ToolBase]] = None,
    ):
        self.agent_fn = agent_fn
        self.estimator_fn = estimator_fn or agent_fn

        selection = build_selection_tools(
            k_simple=k_simple, k_complex=k_complex,
            similarity_fn=similarity_fn or build_backbone_similarity_fn(),
            agent_fn=agent_fn,
        )
        perception_tools = build_perception_tools(
            vlm_fn, ape_host, ape_port, extra_tools=extra_tools
        )
        all_tools = {**selection, **perception_tools}

        self.perception = DisturbanceAwarePerception(
            assess_tool   = selection["assess_quality"],
            select_tool   = selection["select_frames"],
            retrieve_tool = selection["retrieve_frames"],
            tool_library  = all_tools,
            agent_fn      = agent_fn,
        )
        self.synthesis = ConfidenceWeightedSynthesis(agent_fn=agent_fn)

    def run(
        self,
        frames: np.ndarray,
        query: str,
        ground_truth: Optional[str] = None,
        training: bool = False,
    ) -> dict:
        """
        Run Robust-TO on one (video, query) pair.

        training : when True, the optional retrieve-and-retry refinement is
                   disabled so the trajectory is strictly fixed-length for the
                   GRPO reward (Section 3.3).

        Returns a dict: answer, reasoning, reward, n_tool_calls, info.
        """
        # N* is estimated ONCE per question by the frozen estimator (Section 3.3)
        n_opt = estimate_optimal_subqueries(query, agent_fn=self.estimator_fn)

        # ── Disturbance-Aware Adaptive Perception (Section 3.2) ──────────────
        perc: PerceptionOutput = self.perception.run(
            frames=frames, query=query,
            optimal_subqueries=n_opt,
            allow_refinement=not training,
        )

        # ── Confidence-Weighted Evidence Synthesis (Section 3.1) ─────────────
        syn = self.synthesis.run(query, perc.facts)

        # ── GRPO reward on the fixed-length trajectory (Section 3.3) ─────────
        # Reward uses the primary perception call per sub-query (one each), so
        # the trajectory length is fixed regardless of any optional refinement.
        reward = compute_trajectory_reward(
            tool_results       = perc.primary_tool_results,
            answer             = syn.answer,
            ground_truth       = ground_truth,
            n_subqueries       = len(perc.sub_queries),
            optimal_subqueries = n_opt,
            response_text      = syn.raw,
        )

        return {
            "answer":       syn.answer,
            "reasoning":    syn.reasoning,
            "reward":       reward,
            "n_tool_calls": len(perc.all_tool_results),
            "info": {
                "selected_k":       len(perc.selected_indices),
                "pool_size":        len(perc.pool_indices),
                "n_sub_queries":    len(perc.sub_queries),
                "optimal_subq_N*":  n_opt,
                "mean_disturbance": float(perc.disturbance_scores.mean())
                                    if len(perc.disturbance_scores) else 0.0,
            },
        }


# Backward-compatible alias: the method was previously named "RoVid" in code.
RoVidPipeline = RobustTOPipeline


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robust-TO inference pipeline")
    parser.add_argument("--video",      required=True, help="Path to video file")
    parser.add_argument("--question",   required=True, help="Natural language question")
    parser.add_argument("--answer",     default=None,  help="Ground-truth answer (optional)")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    frames, _, video_time = process_video(
        args.video, max_frames=args.max_frames, force_sample=True
    )
    print(f"Loaded {len(frames)} frames from {args.video} ({video_time:.1f}s)")

    pipeline = RobustTOPipeline(agent_fn=agent_fn, vlm_fn=vlm_inference_fn)
    result = pipeline.run(frames, args.question, ground_truth=args.answer)

    print("\n" + "=" * 60)
    print(f"ANSWER:    {result['answer']}")
    print(f"REASONING: {result['reasoning'][:300]}...")
    i = result["info"]
    print("\nPipeline stats:")
    print(f"  Tool calls       : {result['n_tool_calls']}")
    print(f"  Selected K        : {i['selected_k']}   Pool: {i['pool_size']}")
    print(f"  Sub-queries (N*)  : {i['n_sub_queries']} ({i['optimal_subq_N*']})")
    print(f"  Mean disturbance  : {i['mean_disturbance']:.3f}")
    r = result["reward"]
    print("\nReward breakdown (GRPO, Eq. 10):")
    print(f"  R_acc={r.R_acc:.2f}  R_subq={r.R_subq:.2f}  R_cc={r.R_cc:.2f}  R_fmt={r.R_fmt:.2f}")
    print(f"  R_total = {r.R_total:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
