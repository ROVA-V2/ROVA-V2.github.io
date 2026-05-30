"""
RoVid — Stage 1: Tool-Driven Visual Gathering  (Section 0.3 of the paper)

The paper defines exactly TWO stages:

    Stage 1 — Tool-Driven Visual Gathering   (Steps 1.1 – 1.4)
    Stage 2 — Contemplate                    (Loops 2 & 3)

This module provides the unified Stage1 class that encapsulates ALL four steps
of Stage 1 as described in the paper:

    Step 1.1  assess_quality     : parameter-free IQA scoring per frame  (Eq. 2)
    Step 1.2  select_frames      : joint reliability–informativeness rank  (Eq. 3)
    Step 1.3  sub-query decomposition + perception tool dispatch
    Step 1.4  confidence-driven re-invocation                             (Eq. 4)
              → ACTIVE at inference time; SKIPPED when training=True

Internally the implementation delegates Steps 1.1–1.2 to the Scout helper and
Steps 1.3–1.4 to the Perceive helper.  Both helpers are private implementation
details; external code should interact only with Stage1 and Stage1Output.

Inference vs. Training Execution (Section 0.5)
-----------------------------------------------
At inference time the full pipeline runs:
    Steps 1.1 → 1.2 → 1.3 → 1.4  (+ loops 2 & 3 in Stage 2)

During GRPO training each rollout is a SINGLE FORWARD PASS:
    Steps 1.1 → 1.2 → 1.3  only  (no Step 1.4, no Loop 2, no Loop 3)

Pass training=True to enforce this constraint.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..tools.base import ToolBase, ToolResult
from ..tools.selection_tools import AssessQuality, RetrieveFrames, SelectFrames
from .scout import Scout, ScoutOutput
from .perceive import Fact, Perceive, PerceiveOutput


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage1Output:
    """
    Complete output of Stage 1 (Steps 1.1 – 1.4).

    Passed in full to Stage 2 (Contemplate).
    """
    # ── From Steps 1.1–1.2 (Scout) ───────────────────────────────────────────
    selected_frames:    np.ndarray   # (K, H, W, 3) — top-K frames for Stage 2
    selected_indices:   List[int]    # original frame indices
    pool_frames:        np.ndarray   # (N-K, H, W, 3) — retrieval pool P
    pool_indices:       List[int]    # original frame indices of pool
    disturbance_scores: np.ndarray   # per-frame d(fi) ∈ [0,1] for ALL N frames
    # Per-channel disturbance profiles for routing (Section 3.2)
    blur_scores:        np.ndarray   # shape (N,) — per-frame d_blur
    bright_scores:      np.ndarray   # shape (N,) — per-frame d_bright
    occl_scores:        np.ndarray   # shape (N,) — per-frame d_occl

    # ── From Steps 1.3–1.4 (Perceive) ────────────────────────────────────────
    facts:              List[Fact]        # tagged evidence set F = {(ri, ci, src)}
    sub_queries:        List[str]         # sub-queries produced in Step 1.3

    # ── Single-pass subset (for GRPO reward computation) ─────────────────────
    # Reward is always computed on the minimal trajectory (Steps 1.1–1.3 only,
    # no Step 1.4 re-invocations) so that reward is unaffected by the variable-
    # length dynamics of the inference-time loops.  (Section 0.5)
    single_pass_facts:        List[Fact]        = field(default_factory=list)
    single_pass_tool_results: List[ToolResult]  = field(default_factory=list)

    # ── Full tool-call log (all steps including re-invocations) ──────────────
    tool_results: List[ToolResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────

class Stage1:
    """
    Stage 1: Tool-Driven Visual Gathering  (Section 0.3)

    Runs Steps 1.1–1.4 as a single stage.

    Parameters
    ----------
    scout   : Scout  — implements Steps 1.1 (assess_quality) and 1.2 (select_frames)
    perceive: Perceive — implements Steps 1.3 (sub-query decomposition + dispatch)
                         and 1.4 (confidence-driven re-invocation, inference only)
    """

    def __init__(self, scout: Scout, perceive: Perceive) -> None:
        self._scout   = scout
        self._perceive = perceive

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(
        self,
        frames:              np.ndarray,
        query:               str,
        optimal_subqueries:  Optional[int] = None,
        training:            bool = False,
    ) -> Stage1Output:
        """
        Execute Stage 1 (Steps 1.1–1.4).

        Parameters
        ----------
        frames             : (N, H, W, 3) uint8 — all video frames
        query              : natural-language question
        optimal_subqueries : pre-computed N*(q) for sub-query reward; if None,
                             Perceive will estimate it internally
        training           : when True, Step 1.4 (confidence-driven re-invocation)
                             is skipped and the stage executes Steps 1.1–1.3 only,
                             producing a fixed-length rollout suitable for GRPO
                             reward computation  (Section 0.5)

        Returns
        -------
        Stage1Output
        """
        # ── Steps 1.1–1.2: Quality Assessment + Frame Selection ───────────────
        scout_out: ScoutOutput = self._scout.run(frames, query)

        # ── Steps 1.3–1.4: Sub-Query Decomposition, Dispatch, Re-Invocation ───
        # enable_loop1=False when training=True  →  Step 1.4 is skipped,
        # keeping the rollout deterministic (Section 0.5).
        perceive_out: PerceiveOutput = self._perceive.run(
            query              = query,
            selected_frames    = scout_out.selected_frames,
            selected_indices   = scout_out.selected_indices,
            disturbance_scores = scout_out.disturbance_scores,
            pool_frames        = scout_out.pool_frames,
            pool_indices       = scout_out.pool_indices,
            enable_loop1       = not training,   # Step 1.4 active only at inference
            optimal_subqueries = optimal_subqueries,
            blur_scores        = scout_out.blur_scores,
            bright_scores      = scout_out.bright_scores,
            occl_scores        = scout_out.occl_scores,
        )

        return Stage1Output(
            # Steps 1.1–1.2 outputs
            selected_frames    = scout_out.selected_frames,
            selected_indices   = scout_out.selected_indices,
            pool_frames        = scout_out.pool_frames,
            pool_indices       = scout_out.pool_indices,
            disturbance_scores = scout_out.disturbance_scores,
            blur_scores        = scout_out.blur_scores,
            bright_scores      = scout_out.bright_scores,
            occl_scores        = scout_out.occl_scores,
            # Steps 1.3–1.4 outputs
            facts              = perceive_out.facts,
            sub_queries        = perceive_out.sub_queries,
            # Single-pass subset for reward
            single_pass_facts        = perceive_out.single_pass_facts,
            single_pass_tool_results = perceive_out.single_pass_tool_results,
            # All tool results (scout + perceive)
            tool_results = scout_out.tool_results + perceive_out.tool_results,
        )
