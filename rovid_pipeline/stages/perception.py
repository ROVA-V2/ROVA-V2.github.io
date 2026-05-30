"""
Robust-TO — Disturbance-Aware Adaptive Perception  (Section 3.2)

This module implements the feed-forward evidence-acquisition pipeline of Fig. 2,
with NO iterative loops:

    1. assess_quality   — per-frame disturbance profiling  d(fi)        (Eq. 2)
    2. select_frames    — quality-assured reliable frame selection      (Eq. 3)
    3. sub-query decomposition  — Text+Frame, conditioned on the question,
                                  the selected frames' visual content, and the
                                  disturbance profile                   (App. E.1)
    4. disturbance-aware tool routing — two-stage: semantic type -> candidate
                                  tools; dominant corruption -> best candidate
                                  (Table 18, App. E.2)
    5. unified tool call (r, c)  — each tool returns a result tied to its source
                                  frames plus a calibrated confidence           (Eq. 4)

Optional refinement (paper case study, Tab. 7): when a sub-query's initial
evidence is low-confidence and a retrieval pool P is available, retrieve_frames
is dispatched once to pull cleaner frames from P, and the tool is re-invoked on
them (e.g. retrieve_frames(...) -> track_temporal(...) in Tab. 7, sq3).  This is
a single bounded refinement of evidence acquisition, NOT a reasoning loop.

Output: the tagged evidence set  F = {(r_i, c_i, src_i)}  consumed by the
single-pass Confidence-Weighted Evidence Synthesis (stages/synthesis.py, Sec 3.1).
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..tools.base import ToolBase, ToolResult
from ..tools.selection_tools import AssessQuality, RetrieveFrames, SelectFrames
from ..reward import estimate_optimal_subqueries


# Confidence below which an optional single retrieve-and-retry refinement is
# attempted during evidence acquisition (case study, Tab. 7).  Bounded to one
# refinement per sub-query so the trajectory stays well-defined for the reward.
THETA_REFINE = 0.4


@dataclass
class Fact:
    """A single tagged evidence unit  (r_i, c_i, src_i)  (Section 3.1)."""
    sub_query:     str
    result:        Any
    confidence:    float
    source_frames: List[int]
    tool_name:     str
    flagged:       bool = False        # set if a refinement disagreed with the primary
    semantic_type: str = "spatial"     # one of [spatial, temporal, attribute, action, text]
    disturbance:   float = 0.0         # mean d(f) of source frames (for HIGH/MED/LOW tiering)


@dataclass
class PerceptionOutput:
    facts:                List[Fact]
    sub_queries:          List[str]
    selected_indices:     List[int]
    pool_indices:         List[int]
    disturbance_scores:   np.ndarray
    # one ToolResult per sub-query (the primary call) — the fixed-length
    # trajectory used for the GRPO reward (Section 3.3)
    primary_tool_results: List[ToolResult] = field(default_factory=list)
    # every tool call made (assess/select/retrieve/perception) — for stats
    all_tool_results:     List[ToolResult] = field(default_factory=list)


class DisturbanceAwarePerception:
    """
    Disturbance-Aware Adaptive Perception  (Section 3.2).

    Parameters
    ----------
    assess_tool / select_tool / retrieve_tool : the three selection tools.
    tool_library : name -> ToolBase for ALL tools (selection + perception);
                   routing only ever resolves among registered perception tools.
    agent_fn     : fn(prompt) -> str, the host VLM in text-only orchestration mode.
    """

    # Learned routing preferences (Table 18): {semantic_type: {corruption: tool}}.
    # The host VLM routes softly via in-context reasoning (App. E.2); this table
    # is the deterministic fallback used when no agent is available or its output
    # is unparseable.  It mirrors the paper's learned preferences exactly.
    _ROUTING_TABLE: Dict[str, Dict[str, str]] = {
        "spatial":   {"clean": "detect_objects", "blur": "caption_frame",
                      "brightness": "detect_objects", "occlusion": "caption_frame"},
        "attribute": {"clean": "caption_frame", "blur": "caption_frame",
                      "brightness": "caption_frame", "occlusion": "detect_objects"},
        "temporal":  {"clean": "track_temporal", "blur": "recognize_action",
                      "brightness": "track_temporal", "occlusion": "recognize_action"},
        "action":    {"clean": "recognize_action", "blur": "caption_frame",
                      "brightness": "recognize_action", "occlusion": "recognize_action"},
        "text":      {"clean": "read_text", "blur": "caption_frame",
                      "brightness": "read_text", "occlusion": "read_text"},
    }

    # First-stage candidates by semantic type (Section 3.2).
    _CANDIDATES: Dict[str, List[str]] = {
        "spatial":   ["detect_objects", "caption_frame"],
        "attribute": ["caption_frame", "detect_objects"],
        "temporal":  ["track_temporal", "recognize_action"],
        "action":    ["recognize_action", "caption_frame"],
        "text":      ["read_text", "caption_frame"],
    }

    _TOOL_COSTS = {
        "detect_objects": 0.50, "caption_frame": 0.30, "track_temporal": 0.70,
        "recognize_action": 0.60, "read_text": 0.25,
    }

    def __init__(
        self,
        assess_tool:   AssessQuality,
        select_tool:   SelectFrames,
        retrieve_tool: RetrieveFrames,
        tool_library:  Dict[str, ToolBase],
        agent_fn:      Callable[[str], str],
        theta_refine:  float = THETA_REFINE,
    ):
        self.assess        = assess_tool
        self.select        = select_tool
        self.retrieve_tool = retrieve_tool
        self.tools         = tool_library
        self.agent         = agent_fn
        self.theta_refine  = theta_refine

    # ── Public entry-point ────────────────────────────────────────────────────
    def run(
        self,
        frames:             np.ndarray,
        query:              str,
        optimal_subqueries: Optional[int] = None,
        allow_refinement:   bool = True,
    ) -> PerceptionOutput:
        """
        Execute Section 3.2 as a single forward pass.

        allow_refinement : when False, the optional retrieve-and-retry step is
                           disabled so the trajectory is strictly fixed-length
                           (used for GRPO reward computation, Section 3.3).
        """
        N = len(frames)
        all_tool_results: List[ToolResult] = []

        # ── Step 1: assess_quality — per-frame disturbance d(fi) (Eq. 2) ──────
        aq = self.assess(frames, query, None, list(range(N)))
        all_tool_results.append(aq)
        disturbance_scores = aq.result["disturbance_scores"]
        blur_scores   = aq.result["blur_scores"]
        bright_scores = aq.result["bright_scores"]
        occl_scores   = aq.result["occl_scores"]

        # ── Step 2: select_frames — top-K + pool P (Eq. 3) ───────────────────
        sf = self.select(frames, query, disturbance_scores, list(range(N)))
        all_tool_results.append(sf)
        selected_idx = sf.result["selected_indices"]
        pool_idx     = sf.result["pool_indices"]

        sel_frames  = frames[selected_idx] if selected_idx else np.empty((0, *frames.shape[1:]), frames.dtype)
        pool_frames = frames[pool_idx]     if pool_idx     else np.empty((0, *frames.shape[1:]), frames.dtype)

        # ── Averaged disturbance profile d̄ for routing (Section 3.2) ─────────
        def _mean_over(arr, idx):
            return float(np.mean(arr[idx])) if idx and len(arr) > max(idx) else 0.0
        profile = {
            "blur":       _mean_over(blur_scores, selected_idx),
            "brightness": _mean_over(bright_scores, selected_idx),
            "occlusion":  _mean_over(occl_scores, selected_idx),
        }
        dominant = max(profile, key=profile.get)

        # ── Step 3: sub-query decomposition (Text+Frame, App. E.1) ────────────
        typed_sub_queries = self._decompose_query(
            query, optimal_subqueries=optimal_subqueries,
            selected_frames=sel_frames, disturbance_profile=profile,
        )

        sel_dist = disturbance_scores[selected_idx] if selected_idx and \
            len(disturbance_scores) > max(selected_idx) else np.zeros(len(sel_frames))

        # ── Steps 4-5: route + call one tool per sub-query (+ optional refine) ─
        facts: List[Fact] = []
        primary_results: List[ToolResult] = []
        for sq, sq_type in typed_sub_queries:
            tool_name = self._route(sq, sq_type, profile, dominant)
            tr = self._call(tool_name, sel_frames, sq, sel_dist, selected_idx)
            all_tool_results.append(tr)
            primary_results.append(tr)

            fact = self._to_fact(sq, sq_type, tr, disturbance_scores)

            # Optional bounded refinement via retrieve_frames (case study Tab. 7)
            if allow_refinement and tr.confidence < self.theta_refine and len(pool_frames) > 0:
                fact, refine_calls = self._refine(
                    sq, sq_type, tool_name, fact, pool_frames, pool_idx, disturbance_scores
                )
                all_tool_results.extend(refine_calls)

            facts.append(fact)

        return PerceptionOutput(
            facts                = facts,
            sub_queries          = [sq for sq, _ in typed_sub_queries],
            selected_indices     = selected_idx,
            pool_indices         = pool_idx,
            disturbance_scores   = disturbance_scores,
            primary_tool_results = primary_results,
            all_tool_results     = all_tool_results,
        )

    # ── Optional retrieve-and-retry refinement (Tab. 7) ──────────────────────
    def _refine(
        self, sq, sq_type, tool_name, primary_fact,
        pool_frames, pool_idx, all_dist,
    ) -> Tuple[Fact, List[ToolResult]]:
        calls: List[ToolResult] = []
        pool_dist = all_dist[pool_idx] if pool_idx and len(all_dist) > max(pool_idx) \
            else np.zeros(len(pool_frames))

        ret = self.retrieve_tool(pool_frames, sq, pool_dist, pool_idx)
        calls.append(ret)
        if not ret.result or not ret.result.get("retrieved_indices"):
            return primary_fact, calls

        ridx     = ret.result["retrieved_indices"]
        r_frames = pool_frames[ridx]
        r_dist   = pool_dist[ridx]
        r_orig   = [pool_idx[i] for i in ridx]

        retry = self._call(tool_name, r_frames, sq, r_dist, r_orig)
        calls.append(retry)

        # Keep the higher-confidence evidence; flag a disagreement.
        best = retry if retry.confidence > primary_fact.confidence else None
        if best is None:
            return primary_fact, calls
        disagree = (self._norm(retry.result) != "" and
                    self._norm(retry.result) != self._norm(primary_fact.result))
        return self._to_fact(sq, sq_type, retry, all_dist, flagged=disagree), calls

    # ── Sub-query decomposition (App. E.1) ───────────────────────────────────
    def _decompose_query(
        self, query, optimal_subqueries=None,
        selected_frames=None, disturbance_profile=None,
    ) -> List[Tuple[str, str]]:
        target = optimal_subqueries or estimate_optimal_subqueries(query, agent_fn=self.agent)

        # Text+Frame conditioning (App. E.1, Tab. 9): describe the selected
        # frames so decomposition is grounded in visual content, not text alone.
        video_description = "(no visual context available)"
        if selected_frames is not None and len(selected_frames) > 0:
            try:
                video_description = self.agent(
                    f"These are {len(selected_frames)} selected frames from a video. "
                    f"The question is: {query}\n"
                    "Briefly describe (1-2 sentences) the visual elements relevant "
                    "to answering it."
                )
            except Exception:
                video_description = "(visual context extraction failed)"

        dp = disturbance_profile or {"blur": 0.0, "brightness": 0.0, "occlusion": 0.0}
        dist_str = (f"blur={dp.get('blur', 0.0):.2f}, "
                    f"brightness={dp.get('brightness', 0.0):.2f}, "
                    f"occlusion={dp.get('occlusion', 0.0):.2f}")

        prompt = (
            "You are an expert video analyst. Decompose a complex question about "
            "a video into a minimal set of atomic sub-queries. Each sub-query must "
            "target exactly one perceptual primitive and be answerable by a single "
            "visual tool call. Do not generate redundant sub-queries.\n\n"
            "Guidelines:\n"
            "1. Identify the distinct perceptual demands implied by the question.\n"
            "2. For each demand, formulate exactly one atomic sub-query.\n"
            "3. Assign a semantic type: one of [spatial, temporal, attribute, action, text].\n"
            f"4. Minimize the total number of sub-queries. Target about {target}.\n\n"
            f"Input:\n  Video context: {video_description}\n"
            f"  Disturbance profile of selected frames: {dist_str}\n"
            f"  Question: {query}\n\n"
            "Output ONLY a JSON array of objects, no explanation.\n"
            'Example: [{"sub_query": "What objects are near the intersection?", '
            '"type": "spatial"}, {"sub_query": "In what order do they appear?", '
            '"type": "temporal"}]'
        )
        raw = self.agent(prompt)
        valid = {"spatial", "temporal", "attribute", "action", "text"}
        try:
            parsed = json.loads(self._extract_json(raw))
            if isinstance(parsed, list) and parsed:
                pairs = []
                for item in parsed:
                    if isinstance(item, dict):
                        s = str(item.get("sub_query", "")).strip()
                        t = str(item.get("type", "spatial")).strip().lower()
                        t = t if t in valid else "spatial"
                    else:
                        s, t = str(item).strip(), "spatial"
                    if s:
                        pairs.append((s, t))
                if pairs:
                    return pairs[:max(target, 1)]
        except Exception:
            pass
        return [(query, "spatial")]

    # ── Two-stage disturbance-aware routing (Section 3.2, Table 18) ──────────
    def _route(self, sub_query, sq_type, profile, dominant) -> str:
        sq_type = sq_type if sq_type in self._CANDIDATES else "spatial"
        # Stage 1: candidate tools by semantic type, restricted to REGISTERED tools.
        candidates = [c for c in self._CANDIDATES[sq_type] if c in self.tools]
        if not candidates:
            # No registered candidate for this type — fall back to any registered
            # perception tool (keeps a 3-tool release runnable).
            perception = [n for n in self.tools
                          if n not in ("assess_quality", "select_frames", "retrieve_frames")]
            return perception[0] if perception else next(iter(self.tools), "read_text")
        if len(candidates) == 1:
            return candidates[0]

        no_corruption = max(profile.values()) < 1e-6
        corruption = "clean" if no_corruption else dominant

        # Stage 2: let the host VLM pick among candidates under the profile.
        cand_block = "\n".join(
            f"    {c} (cost={self._TOOL_COSTS.get(c, 0.5):.2f})" for c in candidates
        )
        prompt = (
            "You are a tool routing agent. Choose the perception tool that "
            "maximizes reliability under the current corruption.\n"
            "Guidelines:\n"
            "  - spatial under blur: prefer caption_frame over detect_objects.\n"
            "  - temporal under occlusion: prefer recognize_action over track_temporal.\n"
            "  - text under blur: prefer caption_frame over read_text.\n"
            "  - when multiple are viable: prefer lower cost.\n\n"
            f"Sub-query: {sub_query}\n  Semantic type: {sq_type}\n"
            f"  Disturbance: blur={profile['blur']:.2f}, "
            f"brightness={profile['brightness']:.2f}, occlusion={profile['occlusion']:.2f}\n"
            f"  Dominant corruption: {corruption}\n"
            f"  Candidate tools:\n{cand_block}\n\n"
            "Output ONLY the tool name."
        )
        try:
            raw = self.agent(prompt).strip().lower()
            for name in candidates:
                if name in raw:
                    return name
        except Exception:
            pass
        # Deterministic Table-18 fallback, constrained to registered candidates.
        choice = self._ROUTING_TABLE.get(sq_type, {}).get(corruption)
        return choice if choice in candidates else candidates[0]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _call(self, tool_name, frames, sub_query, dist_scores, frame_indices) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(None, 0.0, tool_name, frame_indices)
        return tool(frames, sub_query, dist_scores, frame_indices)

    def _to_fact(self, sq, sq_type, tr, all_dist, flagged=False) -> Fact:
        return Fact(
            sub_query=sq, result=tr.result, confidence=tr.confidence,
            source_frames=tr.source_frames, tool_name=tr.tool_name,
            flagged=flagged, semantic_type=sq_type,
            disturbance=self._source_disturbance(tr.source_frames, all_dist),
        )

    @staticmethod
    def _source_disturbance(source_frames, all_dist) -> float:
        if not source_frames or len(all_dist) == 0:
            return 0.0
        valid = [i for i in source_frames if 0 <= i < len(all_dist)]
        return float(np.mean(all_dist[valid])) if valid else 0.0

    @staticmethod
    def _norm(result) -> str:
        if result is None:
            return ""
        return re.sub(r"\s+", " ", str(result).strip().lower())[:300]

    @staticmethod
    def _extract_json(text) -> str:
        m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        return m.group(0) if m else text
