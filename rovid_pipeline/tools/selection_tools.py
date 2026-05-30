"""
RoVid — Stage 1 Selection Tools

Implements the three selection tools described in Section 3.2:

    assess_quality  : parameter-free IQA scoring per frame   (cost 0.10)
    select_frames   : joint reliability–informativeness rank  (cost 0.15)
    retrieve_frames : disturbance-aware retrieval from pool P (cost 0.20)

All tools share the unified (result, confidence) interface from base.py.

FIXES applied in this file
--------------------------
FIX (Inconsistency 2 — disturbance weights):
    Original code used unequal weights W1=0.5, W2=0.25, W3=0.25.
    The paper (Section C, Eq. 2) explicitly states all three components
    are "weighted equally" after min-max normalisation: w1=w2=w3=1/3.

FIX (Inconsistency 3 — brightness score formula):
    Original code used raw uint8 grayscale mean with a /128 denominator
    and center at 128.0.  The paper (Eq. 12) uses the V-channel of HSV
    space normalised to [0, 1] with center μ_ref=0.5:
        d_bright = 2 * |μ_lum − 0.5|
    Fixed to convert to [0,1] range and use μ_ref=0.5.

FIX (Inconsistency 4 — occlusion score formula):
    Original code used an adaptive threshold (mean + std of gradient)
    and a non-paper *10 scaling factor.  The paper (Eq. 13) uses a fixed
    Sobel-magnitude threshold τ_edge=30 with no additional scaling:
        d_occl = 1 − |{p : G(p) > τ_edge}| / (H×W)

FIX (Inconsistency 5 — select_frames threshold filtering):
    Original code computed s(fi) = (1−d(fi)) * sim without applying the
    paper's indicator-function thresholds (Eq. 3):
        s(fi) = 1(1−d(fi) ≥ θ_rel) · 1(sim ≥ θ_sim) · (1−d(fi)) · sim
    Frames failing either threshold must receive s(fi)=0 and be discarded.
    Added θ_rel=0.55, θ_sim=0.30 as per the paper (Appendix B, Table 15).
"""

from __future__ import annotations
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import ToolBase, ToolResult

SimilarityFn = Callable[[np.ndarray, str], np.ndarray]
AgentFn = Callable[[str], str]

# ── Hyper-parameters from paper (Table 15 / Appendix B) ──────────────────────
TAU_BLUR   = 500.0   # Laplacian normalisation constant  (Sec. C, Eq. 11)
MU_REF     = 0.5     # neutral luminance midpoint         (Sec. C, Eq. 12)
TAU_EDGE   = 30.0    # Sobel edge threshold               (Sec. C, Eq. 13)
THETA_REL  = 0.55    # reliability threshold for select_frames  (Table 15)
THETA_SIM  = 0.30    # relevance  threshold for select_frames   (Table 15)


# ── Disturbance sub-metrics ───────────────────────────────────────────────────

def _blur_score(frame: np.ndarray) -> float:
    """
    d_blur from Eq. 11:  d_blur = 1 − min(1, Var(∇²f) / τ_blur)

    Sharp frames have high Laplacian variance → low d_blur.
    """
    gray = _to_gray(frame)
    lap_var = _laplacian_variance(gray)
    sharpness = min(lap_var / TAU_BLUR, 1.0)
    return float(1.0 - sharpness)


def _brightness_score(frame: np.ndarray) -> float:
    """
    d_bright from Eq. 12:  d_bright = 2 * |μ_lum(fi) − μ_ref|

    where μ_lum ∈ [0, 1] is the mean pixel intensity in the V channel
    of HSV space and μ_ref = 0.5 (both under- and over-exposed penalised).

    FIX: Original code used raw uint8 mean /128 with centre 128.0.
    Paper uses [0,1]-normalised value with centre 0.5.
    """
    gray = _to_gray(frame)
    # Normalise to [0,1] (equivalent to HSV V channel for grayscale)
    # Clip normalised mean to [0,1] before applying formula to handle
    # floating-point imprecision from luminance coefficient rounding (0.2989+
    # 0.5870+0.1140=0.9999 ≠ 1.0).
    mu_lum = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
    d = 2.0 * abs(mu_lum - MU_REF)
    return float(np.clip(d, 0.0, 1.0))


def _occlusion_score(frame: np.ndarray) -> float:
    """
    d_occl from Eq. 13:
        d_occl = 1 − |{p : G(p) > τ_edge}| / (H×W)

    where G(p) = sqrt(Gx(p)² + Gy(p)²) is the Sobel gradient magnitude
    and τ_edge=30.

    FIX (Inconsistency 4b — gradient operator):
        Previous code used np.diff (simple [1, -1] finite differences)
        instead of proper Sobel filters.  The paper (Eq. 13, Sec. C)
        specifies Sobel gradient magnitude, which uses 3×3 convolution
        kernels [-1,0,1; -2,0,2; -1,0,1] and its transpose.  Sobel
        produces stronger, more noise-tolerant gradients than np.diff,
        which affects the edge pixel count and thus the occlusion score.
    """
    gray = _to_gray(frame)
    H, W = gray.shape
    # Sobel kernels (3×3)
    # Gx kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Gy kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -1.0 * padded[:-2, :-2] + 1.0 * padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
        - 1.0 * padded[2:, :-2] + 1.0 * padded[2:, 2:]
    )
    gy = (
        -1.0 * padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - 1.0 * padded[:-2, 2:]
        + 1.0 * padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + 1.0 * padded[2:, 2:]
    )
    G = np.hypot(gx, gy)
    edge_pixel_count = float(np.sum(G > TAU_EDGE))
    edge_fraction = edge_pixel_count / max(H * W, 1)
    return float(np.clip(1.0 - edge_fraction, 0.0, 1.0))


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    rgb = frame[..., :3].astype(np.float32)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lap = (
        padded[:-2, 1:-1] +
        padded[2:, 1:-1] +
        padded[1:-1, :-2] +
        padded[1:-1, 2:] -
        4.0 * center
    )
    return float(np.var(lap))


# ── assess_quality ────────────────────────────────────────────────────────────

class AssessQuality(ToolBase):
    """
    Parameter-free IQA scoring per frame.

    Eq. 2:  d(fi) = d_blur(fi) + d_bright(fi) + d_occl(fi)
            (all three terms weighted equally after per-video min-max
            normalisation, then average — Section C)

    FIX: Original code used unequal weights (0.5, 0.25, 0.25).
    Paper Section C states "weighted equally" → W1=W2=W3=1/3.

    Returns
    -------
    result : dict with keys
        'disturbance_scores' : np.ndarray of shape (N,)  — per-frame d(fi) ∈ [0,1]
        'blur_scores'        : np.ndarray  — per-frame d_blur
        'bright_scores'      : np.ndarray  — per-frame d_bright
        'occl_scores'        : np.ndarray  — per-frame d_occl
    """

    name = "assess_quality"

    # Equal weights as stated in paper Section C
    W_BLUR   = 1.0 / 3.0
    W_BRIGHT = 1.0 / 3.0
    W_OCCL   = 1.0 / 3.0

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Dict, float]:
        N = len(frames)
        blur_scores   = np.array([_blur_score(f)       for f in frames], dtype=np.float32)
        bright_scores = np.array([_brightness_score(f) for f in frames], dtype=np.float32)
        occl_scores   = np.array([_occlusion_score(f)  for f in frames], dtype=np.float32)

        # Per-video min-max normalisation before combining (Section C)
        blur_n   = _minmax_norm(blur_scores)
        bright_n = _minmax_norm(bright_scores)
        occl_n   = _minmax_norm(occl_scores)

        scores = np.clip(
            self.W_BLUR * blur_n + self.W_BRIGHT * bright_n + self.W_OCCL * occl_n,
            0.0, 1.0
        ).astype(np.float32)

        # c_intrinsic = 1.0 (deterministic tool; no model uncertainty — Section B.1)
        c_intrinsic = 1.0

        return {
            "disturbance_scores": scores,
            "blur_scores":        blur_scores,
            "bright_scores":      bright_scores,
            "occl_scores":        occl_scores,
        }, c_intrinsic


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1]; returns zeros if range is zero."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── select_frames ─────────────────────────────────────────────────────────────

class SelectFrames(ToolBase):
    """
    Joint reliability–informativeness ranking.

    Eq. 3:
        s(fi) = 1(1−d(fi) ≥ θ_rel) · 1(sim(φ(fi),ψ(q)) ≥ θ_sim)
              · (1−d(fi)) · sim(φ(fi), ψ(q))

    Frames failing either threshold receive s(fi)=0 and are discarded
    (not forwarded to Stage 1 Step 1.3).

    FIX (Inconsistency 5): Original code computed scores = reliability * similarity
    without applying the indicator-function thresholds.  This allowed heavily
    corrupted but query-relevant frames to rank in the top-K.  The paper is
    explicit: "Frames that fail either threshold receive s(fi)=0 and are discarded."

    K is determined dynamically in [4, 12] by the host VLM (or heuristic fallback).
    """

    name = "select_frames"

    def __init__(
        self,
        k_simple: int = 4,
        k_complex: int = 12,
        complexity_threshold: int = 10,
        theta_rel: float = THETA_REL,
        theta_sim: float = THETA_SIM,
        similarity_fn: Optional[SimilarityFn] = None,
        agent_fn: Optional[AgentFn] = None,
    ):
        self.k_simple = k_simple
        self.k_complex = k_complex
        self.complexity_threshold = complexity_threshold
        self.theta_rel = theta_rel
        self.theta_sim = theta_sim
        self.similarity_fn = similarity_fn
        self.agent_fn = agent_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Dict, float]:
        N = len(frames)
        if N == 0:
            return None, 0.0

        K = self._select_k(sub_query, N)

        similarities = self._compute_similarities(frames, sub_query)
        reliability  = 1.0 - disturbance_scores   # (N,)

        # Eq. 3 — indicator-function thresholds (FIX: was missing entirely)
        reliable_mask  = reliability  >= self.theta_rel   # 1(1−d ≥ θ_rel)
        relevant_mask  = similarities >= self.theta_sim   # 1(sim ≥ θ_sim)
        eligible_mask  = reliable_mask & relevant_mask

        # Score = 0 for ineligible frames; multiplicative for eligible
        scores = np.where(eligible_mask, reliability * similarities, 0.0).astype(np.float32)

        ranked_indices   = np.argsort(scores)[::-1]           # descending
        selected_indices = sorted(ranked_indices[:K].tolist())
        pool_indices     = sorted(ranked_indices[K:].tolist())

        selected_scores = scores[ranked_indices[:K]]
        c_intrinsic = float(selected_scores.mean()) if len(selected_scores) > 0 else 0.0
        c_intrinsic = float(np.clip(c_intrinsic, 0.0, 1.0))

        return {
            "selected_indices": selected_indices,
            "pool_indices":     pool_indices,
            "scores":           scores.tolist(),
        }, c_intrinsic

    def _compute_similarities(self, frames: np.ndarray, sub_query: str) -> np.ndarray:
        """
        sim(φ(fi), ψ(q)) with the shared VLM backbone when available.
        Falls back to neutral score (1.0) when no encoder is provided.
        """
        if self.similarity_fn is None:
            return np.ones(len(frames), dtype=np.float32)

        try:
            similarities = np.asarray(self.similarity_fn(frames, sub_query), dtype=np.float32)
        except Exception:
            similarities = np.ones(len(frames), dtype=np.float32)

        if similarities.shape != (len(frames),):
            similarities = np.ones(len(frames), dtype=np.float32)

        # Shift cosine similarities from [-1,1] to [0,1]
        similarities = (similarities + 1.0) / 2.0
        return np.clip(similarities, 0.0, 1.0)

    def _select_k(self, sub_query: str, n_frames: int) -> int:
        """
        K is adapted by the LLM to query complexity (Section 3.2).
        Falls back to a word-count heuristic when no agent is available.
        K is clipped to [k_simple, k_complex] = [4, 12] as per paper.
        """
        if self.agent_fn is not None:
            prompt = (
                f"Question: {sub_query}\n"
                f"Available frames: {n_frames}\n\n"
                "Choose how many frames K should be selected for visual reasoning. "
                f"Return ONLY a JSON object like {{\"k\": {self.k_simple}}}. "
                f"Prefer values between {self.k_simple} and {min(self.k_complex, n_frames)}. "
                "Use smaller K for single-hop factual questions and larger K for temporal or multi-hop questions."
            )
            try:
                raw = self.agent_fn(prompt)
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    value = int(parsed.get("k", 0))
                    if value > 0:
                        return min(max(value, self.k_simple), min(self.k_complex, n_frames))
            except Exception:
                pass

        word_count = len(sub_query.split())
        heuristic_k = self.k_complex if word_count >= self.complexity_threshold else self.k_simple
        return min(heuristic_k, n_frames)


# ── retrieve_frames ───────────────────────────────────────────────────────────

class RetrieveFrames(ToolBase):
    """
    Disturbance-aware retrieval from pool P.

    Given the pool P assembled during Scout and a refined query q',
    retrieves the frames most relevant AND least disturbed (Eq. 5 / Loop 3).
    """

    name = "retrieve_frames"

    def __init__(self, top_k: int = 4, similarity_fn: Optional[SimilarityFn] = None):
        self.top_k = top_k
        self.similarity_fn = similarity_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Dict, float]:
        N = len(frames)
        if N == 0:
            return None, 0.0

        # Re-use SelectFrames ranking with fixed K (no agent needed for retrieval)
        selector = SelectFrames(
            k_simple=self.top_k,
            k_complex=self.top_k,
            similarity_fn=self.similarity_fn,
        )
        inner = selector._run(frames, sub_query, disturbance_scores)
        if inner[0] is None:
            return None, 0.0

        result_dict, c_intrinsic = inner
        return {
            "retrieved_indices": result_dict["selected_indices"],
            "scores": result_dict["scores"],
        }, c_intrinsic


# ── convenience factory ───────────────────────────────────────────────────────

def build_selection_tools(
    k_simple: int = 4,
    k_complex: int = 12,
    similarity_fn: Optional[SimilarityFn] = None,
    agent_fn: Optional[AgentFn] = None,
) -> Dict[str, ToolBase]:
    return {
        "assess_quality":  AssessQuality(),
        "select_frames":   SelectFrames(
            k_simple=k_simple,
            k_complex=k_complex,
            similarity_fn=similarity_fn,
            agent_fn=agent_fn,
        ),
        "retrieve_frames": RetrieveFrames(similarity_fn=similarity_fn),
    }
