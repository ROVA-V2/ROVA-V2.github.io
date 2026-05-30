"""
RoVid — Stage 1 internal helper: Steps 1.1 & 1.2  (Section 0.3 of the paper)

The paper describes a single Stage 1 ("Tool-Driven Visual Gathering") containing
Steps 1.1 – 1.4.  Scout implements only the first two steps:

    Step 1.1  assess_quality  — parameter-free IQA scoring per frame    (Eq. 2)
    Step 1.2  select_frames   — joint reliability–informativeness rank    (Eq. 3)

Steps 1.3 and 1.4 (sub-query decomposition, tool dispatch, and confidence-driven
re-invocation) are implemented in perceive.py.

External code should NOT instantiate Scout directly.  Use the Stage1 class in
stages/stage1.py, which orchestrates both Scout and Perceive under the unified
Stage 1 interface described in the paper.

Outputs
-------
ScoutOutput
    selected_frames       : (K, H, W, 3) top-K frames for Step 1.3
    selected_indices      : original frame indices of selected frames
    pool_frames           : (N-K, H, W, 3) retrieval pool P for Loop 3 look-back
    pool_indices          : original frame indices of pool frames
    disturbance_scores    : per-frame d(fi) for ALL N frames  — propagated to
                            Eq. 1 confidence modulation for all downstream tools
    quality_profile       : alias for disturbance_scores (paper terminology)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..tools.base import ToolResult
from ..tools.selection_tools import AssessQuality, SelectFrames


@dataclass
class ScoutOutput:
    # Top-K frames → forwarded to Perceive
    selected_frames:   np.ndarray
    selected_indices:  List[int]
    # Pool P → retained for Contemplate's Loop 3 look-back
    pool_frames:       np.ndarray
    pool_indices:      List[int]
    # Quality profile — propagates to Eq. 1 confidence for ALL downstream tools
    disturbance_scores: np.ndarray       # shape (N,) for the full video
    quality_profile:    np.ndarray       # alias (same object)
    # Per-channel disturbance profiles for disturbance-aware tool routing (Sec 3.2)
    blur_scores:        np.ndarray       # shape (N,) — per-frame d_blur
    bright_scores:      np.ndarray       # shape (N,) — per-frame d_bright
    occl_scores:        np.ndarray       # shape (N,) — per-frame d_occl
    # Tool call records for reward computation
    tool_results:      List[ToolResult] = field(default_factory=list)


class Scout:
    """
    Stage 1 internal helper — Steps 1.1 & 1.2: Disturbance-Aware Frame Selection

    Step 1.1  assess_quality  → compute d(fi) for every frame          (Eq. 2)
    Step 1.2  select_frames   → joint reliability-informativeness rank  (Eq. 3)
               Top-K  → selected_frames (forwarded to Steps 1.3–1.4)
               Rest   → pool P          (retained for Stage 2 / Loop 3 look-back)

    The quality profile {d(fi)} produced here:
      • parameterises the confidence modulation in Eq. 1 for all subsequent tools
      • guides Loop 3 retrieval toward low-disturbance evidence

    NOTE: Instantiate via Stage1 (stages/stage1.py), not directly.
    """

    def __init__(
        self,
        assess_tool: AssessQuality,
        select_tool: SelectFrames,
    ):
        self.assess = assess_tool
        self.select = select_tool

    def run(self, frames: np.ndarray, query: str) -> ScoutOutput:
        """
        Parameters
        ----------
        frames : (N, H, W, 3) uint8 — all video frames
        query  : natural language question
        """
        N = len(frames)
        tool_results: List[ToolResult] = []

        # ── Step 1: Quality Assessment ────────────────────────────────────────
        # assess_quality is called with no prior disturbance scores (cold start)
        aq_result = self.assess(
            frames=frames,
            sub_query=query,
            disturbance_scores=None,
            frame_indices=list(range(N)),
        )
        tool_results.append(aq_result)

        disturbance_scores: np.ndarray = aq_result.result["disturbance_scores"]

        # ── Step 2: Joint Reliability–Informativeness Ranking ─────────────────
        sf_result = self.select(
            frames=frames,
            sub_query=query,
            disturbance_scores=disturbance_scores,
            frame_indices=list(range(N)),
        )
        tool_results.append(sf_result)

        selected_idx: List[int] = sf_result.result["selected_indices"]
        pool_idx:     List[int] = sf_result.result["pool_indices"]

        selected_frames = frames[selected_idx]   if selected_idx else np.empty((0, *frames.shape[1:]))
        pool_frames     = frames[pool_idx]       if pool_idx     else np.empty((0, *frames.shape[1:]))

        return ScoutOutput(
            selected_frames=selected_frames,
            selected_indices=selected_idx,
            pool_frames=pool_frames,
            pool_indices=pool_idx,
            disturbance_scores=disturbance_scores,
            quality_profile=disturbance_scores,
            blur_scores=aq_result.result["blur_scores"],
            bright_scores=aq_result.result["bright_scores"],
            occl_scores=aq_result.result["occl_scores"],
            tool_results=tool_results,
        )
