"""
RoVid Pipeline — Main Entry Point  (Section 0 of the paper)

Orchestrates the TWO stages described in the paper:

    Stage 1 — Tool-Driven Visual Gathering   (Steps 1.1–1.4)
              assess_quality → select_frames → sub-query decomposition
              → perception tool dispatch → confidence-driven re-invocation

    Stage 2 — Contemplate                    (Loops 2 & 3)
              confidence-weighted reasoning → gap-driven active inquiry
              → uncertainty-triggered look-back

Inference vs. Training
----------------------
At inference time both stages run fully including all loops.
During GRPO training pass training=True to RoVidPipeline.run():
    Stage 1 executes Steps 1.1–1.3 only (no Step 1.4 re-invocation)
    Stage 2 executes a single reasoning pass (no Loop 2, no Loop 3)
This matches the paper's constraint (Section 0.5): rewards are computed on a
minimal, fixed-length trajectory so that reward is unaffected by the variable-
length dynamics of the inference-time loops.

Usage
-----
    python rovid_pipeline.py
        --video /path/to/video.mp4
        --question "What is the person doing at the beginning?"

For integration into other VLMs, override:
    1. The model checkpoint and load function (lines marked # [REPLACE MODEL])
    2. The llava_inference() function (lines marked # [REPLACE INFERENCE])
    3. Optionally: process_video() for model-specific frame pre-processing
"""

from __future__ import annotations
import argparse
import copy
import os
from typing import Callable, Optional

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

from .stages.contemplate import Contemplate, ContemplateOutput
from .stages.perceive import Perceive
from .stages.scout import Scout
from .stages.stage1 import Stage1, Stage1Output
from .tools.perception_tools import build_perception_tools
from .tools.selection_tools import build_selection_tools
from .reward import compute_trajectory_reward, estimate_optimal_subqueries


# ─────────────────────────────────────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────────────────────────────────────

MAX_FRAMES = 32


def process_video(
    video_path: str,
    max_frames: int = MAX_FRAMES,
    fps: int = 1,
    force_sample: bool = False,
) -> tuple[np.ndarray, str, float]:
    """Returns (frames [N,H,W,3], frame_time_str, video_duration_sec)."""
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
        frames = vr.get_batch(frame_idx).asnumpy()   # (N, H, W, 3)
        return frames, time_str, video_time

    if cv2 is None:
        raise RuntimeError("Neither decord nor OpenCV is installed; process_video cannot read videos")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 1e-6)
    if total <= 0:
        frames_list = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        total = len(frames_list)
        if total == 0:
            raise RuntimeError(f"Video contains no readable frames: {video_path}")
        all_frames = np.stack(frames_list, axis=0)
    else:
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

# Load once at module import (heavy; skip during testing by setting env var)
if os.environ.get("ROVID_SKIP_MODEL_LOAD") != "1" and load_pretrained_model is not None and torch is not None:
    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            MODEL_NAME,
            None,
            "llava_qwen",
            torch_dtype="bfloat16",
            device_map="auto",
            overwrite_config={},
        )
        model.eval()
    except Exception:
        tokenizer = model = image_processor = max_length = None
else:
    tokenizer = model = image_processor = max_length = None   # test mode


def llava_inference(prompt: str, frames: Optional[np.ndarray]) -> str:
    """
    Call the VLM with an optional video clip.

    Parameters
    ----------
    prompt : text prompt
    frames : (N, H, W, 3) uint8 or None (text-only call)
    """
    if model is None or tokenizer is None or image_processor is None:
        raise RuntimeError(
            "LLaVA model is unavailable. Install the LLaVA dependency or unset "
            "ROVID_SKIP_MODEL_LOAD only in an environment where the model can load."
        )

    # [REPLACE INFERENCE] — adapt for your VLM
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
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=video_tensor,
            modalities=modalities,
            do_sample=False,
            temperature=0,
            max_new_tokens=1024,
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
    Reuse the loaded VLM backbone for Eq. 3 instead of loading a separate CLIP model.
    Falls back to neutral similarity scores when the underlying model does not expose
    compatible vision/text embeddings.
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
        text_model = None
        if hasattr(model, "get_model"):
            try:
                text_model = model.get_model()
            except Exception:
                text_model = None
        embed_tokens = getattr(text_model, "embed_tokens", None) if text_model is not None else None
        if embed_tokens is None:
            embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed_tokens is None:
            return None
        with torch.no_grad():
            embeds = embed_tokens(input_ids)
        pooled = embeds.mean(dim=1)
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _encode_images(frames: np.ndarray):
        if not hasattr(model, "get_vision_tower"):
            return None
        try:
            vision_tower = model.get_vision_tower()
        except Exception:
            return None
        if vision_tower is None:
            return None

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        try:
            pixel_values = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            with torch.no_grad():
                image_outputs = vision_tower(pixel_values)
            hidden = getattr(image_outputs, "last_hidden_state", image_outputs)
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[0]
            if hidden is None:
                return None
            pooled = hidden.mean(dim=1)
            return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        except Exception:
            return None

    def _similarity(frames: np.ndarray, query: str) -> np.ndarray:
        text_feat = _encode_text(query)
        image_feat = _encode_images(frames)
        if text_feat is None or image_feat is None:
            return _neutral(frames, query)
        text_feat = text_feat.to(image_feat.device, dtype=image_feat.dtype)
        sims = torch.matmul(image_feat, text_feat.transpose(0, 1)).squeeze(-1)
        return sims.detach().cpu().float().numpy()

    return _similarity


# ─────────────────────────────────────────────────────────────────────────────
# RoVid pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RoVidPipeline:
    """
    Full two-stage RoVid pipeline  (Section 0 of the paper).

    Stage 1 — Tool-Driven Visual Gathering (Steps 1.1–1.4)
    Stage 2 — Contemplate: Reasoning with Active Inquiry (Loops 2 & 3)

    Every visual tool reports not only what it sees, but how much it trusts
    what it sees.  This turns tool orchestration from blind delegation into
    informed reasoning under uncertainty.
    """

    def __init__(
        self,
        agent_fn,           # text-only LLM call
        vlm_fn,             # vision+language call
        ape_host:  str = "0.0.0.0",
        ape_port:  int = 9999,
        k_simple:  int = 4,
        k_complex: int = 12,
        similarity_fn: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
    ):
        # ── Build tool library ────────────────────────────────────────────────
        selection  = build_selection_tools(
            k_simple=k_simple,
            k_complex=k_complex,
            similarity_fn=similarity_fn or build_backbone_similarity_fn(),
            agent_fn=agent_fn,
        )
        perception = build_perception_tools(vlm_fn, ape_host, ape_port)
        all_tools  = {**selection, **perception}
        self.agent_fn = agent_fn

        # ── Stage 1 internal substep helpers ─────────────────────────────────
        _scout = Scout(
            assess_tool = selection["assess_quality"],
            select_tool = selection["select_frames"],
        )
        _perceive = Perceive(
            tool_library    = all_tools,
            agent_fn        = agent_fn,
            retrieve_frames = selection["retrieve_frames"],
        )

        # ── Instantiate the two paper stages ─────────────────────────────────
        # Stage 1: Tool-Driven Visual Gathering (Steps 1.1–1.4)
        self.stage1 = Stage1(scout=_scout, perceive=_perceive)

        # Stage 2: Contemplate — Reasoning with Active Inquiry (Loops 2 & 3)
        self.contemplate = Contemplate(
            agent_fn        = agent_fn,
            perceive_stage  = _perceive,
            retrieve_tool   = selection["retrieve_frames"],
        )

    def run(
        self,
        frames:       np.ndarray,
        query:        str,
        ground_truth: Optional[str] = None,
        training:     bool = False,
    ) -> dict:
        """
        Run the two-stage RoVid pipeline.

        Parameters
        ----------
        frames       : (N, H, W, 3) uint8 — all video frames
        query        : natural language question
        ground_truth : optional correct answer for reward computation
        training     : when True, enforces the GRPO training constraint
                       (Section 0.5): Stage 1 executes Steps 1.1–1.3 only
                       (no Step 1.4) and Stage 2 executes a single reasoning
                       pass (no Loop 2, no Loop 3).  Reward is computed on
                       this minimal, fixed-length trajectory.

        Returns
        -------
        dict with keys: answer, reasoning, reward, stage1_info, n_tool_calls
        """
        all_tool_results = []

        # ── Stage 1: Tool-Driven Visual Gathering (Steps 1.1–1.4) ─────────────
        # training=True → Steps 1.1–1.3 only (Step 1.4 disabled)
        optimal_subqueries = estimate_optimal_subqueries(query, agent_fn=self.agent_fn)
        stage1_out: Stage1Output = self.stage1.run(
            frames             = frames,
            query              = query,
            optimal_subqueries = optimal_subqueries,
            training           = training,
        )
        all_tool_results.extend(stage1_out.tool_results)

        # ── Reward trajectory: single-pass facts from Stage 1 (Section 0.5) ──
        # Reward is computed on Steps 1.1–1.3 only (no re-invocations), so that
        # the reward signal is unaffected by the variable-length loop dynamics.
        # When training=True, stage1_out.facts IS the single-pass output already.
        reward_facts        = stage1_out.single_pass_facts if not training else stage1_out.facts
        reward_tool_results = stage1_out.single_pass_tool_results if not training \
                              else [tr for tr in stage1_out.tool_results]

        training_pass = self.contemplate.reason_once(query, reward_facts)

        # ── Stage 2: Contemplate — Reasoning with Active Inquiry ──────────────
        # training=True → single reasoning pass (no Loop 2, no Loop 3)
        contemplate_out: ContemplateOutput = self.contemplate.run(
            query              = query,
            facts              = stage1_out.facts,
            pool_frames        = stage1_out.pool_frames,
            pool_indices       = stage1_out.pool_indices,
            disturbance_scores = stage1_out.disturbance_scores,
            selected_frames    = stage1_out.selected_frames,
            selected_indices   = stage1_out.selected_indices,
            training           = training,
        )
        all_tool_results.extend(contemplate_out.tool_results)

        # ── Reward computation (GRPO, Section 0.5) ───────────────────────────
        n_subqueries = len(stage1_out.sub_queries)
        reward = compute_trajectory_reward(
            tool_results        = reward_tool_results,
            answer              = self.contemplate._extract_answer(training_pass.answer),
            ground_truth        = ground_truth,
            n_subqueries        = n_subqueries,
            optimal_subqueries  = optimal_subqueries,
            response_text       = training_pass.raw_response,
        )

        return {
            "answer":       contemplate_out.answer,
            "reasoning":    contemplate_out.reasoning,
            "reward":       reward,
            "n_tool_calls": len(all_tool_results),
            "loop2_iters":  contemplate_out.loop2_iters,
            "loop3_iters":  contemplate_out.loop3_iters,
            "stage1_info": {
                "selected_k":       len(stage1_out.selected_indices),
                "pool_size":        len(stage1_out.pool_indices),
                "n_sub_queries":    n_subqueries,
                "mean_disturbance": float(stage1_out.disturbance_scores.mean()),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoVid inference pipeline")
    parser.add_argument("--video",    required=True,  help="Path to video file")
    parser.add_argument("--question", required=True,  help="Natural language question")
    parser.add_argument("--answer",   default=None,   help="Ground-truth answer (optional)")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    # Load video
    frames, frame_time, video_time = process_video(
        args.video, max_frames=args.max_frames, force_sample=True
    )
    print(f"Loaded {len(frames)} frames from {args.video} ({video_time:.1f}s)")

    # Run pipeline
    pipeline = RoVidPipeline(agent_fn=agent_fn, vlm_fn=vlm_inference_fn)
    result   = pipeline.run(frames, args.question, ground_truth=args.answer)

    print("\n" + "="*60)
    print(f"ANSWER:     {result['answer']}")
    print(f"REASONING:  {result['reasoning'][:300]}...")
    print(f"\nPipeline stats:")
    print(f"  Tool calls      : {result['n_tool_calls']}")
    print(f"  Loop-2 iters    : {result['loop2_iters']}")
    print(f"  Loop-3 iters    : {result['loop3_iters']}")
    s = result['stage1_info']
    print(f"  Stage 1: K={s['selected_k']}, Pool={s['pool_size']}, "
          f"Sub-queries={s['n_sub_queries']}, "
          f"Mean disturbance={s['mean_disturbance']:.3f}")
    print(f"\nReward breakdown (for GRPO):")
    r = result["reward"]
    print(f"  R_acc={r.R_acc:.2f}  R_subq={r.R_subq:.2f}  "
          f"R_cc={r.R_cc:.2f}  R_fmt={r.R_fmt:.2f}")
    print(f"  R_total = {r.R_total:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
