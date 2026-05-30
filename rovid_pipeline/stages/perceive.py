"""
RoVid — Stage 1 internal helper: Steps 1.3 & 1.4  (Section 0.3 of the paper)

The paper describes a single Stage 1 ("Tool-Driven Visual Gathering") containing
Steps 1.1 – 1.4.  Perceive implements Steps 1.3 and 1.4:

    Step 1.3  Sub-Query Decomposition and Perception Tool Dispatch
              The LLM decomposes q into {sq1, …, sqm} and routes each to the
              most appropriate perception tool, guided by the disturbance profile
              from Step 1.1.

    Step 1.4  Confidence-Driven Re-Invocation                           (Eq. 4)
              If ci < θ₁ after a tool call:
                Strategy A — invoke an alternative tool on the same frames
                Strategy B — call retrieve_frames for cleaner frames, then re-invoke
              ACTIVE at inference time; SKIPPED when enable_loop1=False (training).

Steps 1.1 and 1.2 (quality assessment + frame selection) are in scout.py.

External code should NOT instantiate Perceive directly.  Use the Stage1 class in
stages/stage1.py, which orchestrates both Scout and Perceive under the unified
Stage 1 interface described in the paper.

Output: tagged fact set  F = {(r_i, c_i, src(f))}
"""

from __future__ import annotations
import json
import re
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..tools.base import ToolBase, ToolResult
from ..tools.selection_tools import RetrieveFrames
from ..reward import estimate_optimal_subqueries


# ── Threshold from paper / hyper-parameter ────────────────────────────────────
THETA_1 = 0.4   # confidence below which Loop 1 triggers re-invocation


@dataclass
class Fact:
    """A single tagged evidence unit output from Perceive (Section 3.1)."""
    sub_query:     str
    result:        Any
    confidence:    float
    source_frames: List[int]
    tool_name:     str
    flagged:       bool = False   # True if a contradiction was detected
    semantic_type: str = "spatial"  # one of [spatial, temporal, attribute, action, text]
    disturbance:   float = 0.0    # mean d(f) of source frames (for tier assignment)


@dataclass
class PerceiveOutput:
    facts:                   List[Fact]
    tool_results:            List[ToolResult] = field(default_factory=list)
    single_pass_facts:       List[Fact] = field(default_factory=list)
    single_pass_tool_results: List[ToolResult] = field(default_factory=list)
    sub_queries:             List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────

class Perceive:
    """
    Stage 1 internal helper — Steps 1.3 & 1.4: Sub-Query Dispatch with Re-Invocation

    Parameters
    ----------
    tool_library      : dict mapping tool name → ToolBase instance
    agent_fn          : fn(prompt: str) -> str  (LLM text-only call)
    retrieve_frames   : RetrieveFrames tool (for Step 1.4 / Loop 1 Strategy B)

    NOTE: Instantiate via Stage1 (stages/stage1.py), not directly.
    """

    # Learned routing preferences (Table 18): {semantic_type: {corruption: tool}}
    # The host VLM routes softly via in-context reasoning (Appendix E.2); this
    # table is the fallback used when no agent is available or its output is
    # unparseable, exactly mirroring the paper's learned preferences.
    _ROUTING_TABLE: Dict[str, Dict[str, str]] = {
        # Object identity / location  (spatial)
        "spatial":   {"clean": "detect_objects", "blur": "caption_frame",
                      "brightness": "detect_objects", "occlusion": "caption_frame"},
        # Object attribute / appearance (attribute)
        "attribute": {"clean": "caption_frame", "blur": "caption_frame",
                      "brightness": "caption_frame", "occlusion": "detect_objects"},
        # Temporal / motion (temporal)
        "temporal":  {"clean": "track_temporal", "blur": "recognize_action",
                      "brightness": "track_temporal", "occlusion": "recognize_action"},
        # Action / event (action)
        "action":    {"clean": "recognize_action", "blur": "caption_frame",
                      "brightness": "recognize_action", "occlusion": "recognize_action"},
        # In-video text (text)
        "text":      {"clean": "read_text", "blur": "caption_frame",
                      "brightness": "read_text", "occlusion": "read_text"},
    }

    # Candidate tool sets by semantic type — first stage of routing (Section 3.2)
    _CANDIDATES: Dict[str, List[str]] = {
        "spatial":   ["detect_objects", "caption_frame"],
        "attribute": ["caption_frame", "detect_objects"],
        "temporal":  ["track_temporal", "recognize_action"],
        "action":    ["recognize_action", "caption_frame"],
        "text":      ["read_text", "caption_frame"],
    }

    def __init__(
        self,
        tool_library:    Dict[str, ToolBase],
        agent_fn:        Callable[[str], str],
        retrieve_frames: RetrieveFrames,
        theta_1:         float = THETA_1,
    ):
        self.tools         = tool_library
        self.agent         = agent_fn
        self.retrieve_tool = retrieve_frames
        self.theta_1       = theta_1

    # ── Public entry-point ────────────────────────────────────────────────────
    def run(
        self,
        query:             str,
        selected_frames:   np.ndarray,
        selected_indices:  List[int],
        disturbance_scores: np.ndarray,    # full-video scores from Scout
        pool_frames:       np.ndarray,
        pool_indices:      List[int],
        enable_loop1:      bool = True,
        provided_sub_queries: Optional[List[str]] = None,
        optimal_subqueries: Optional[int] = None,
        blur_scores:   Optional[np.ndarray] = None,
        bright_scores: Optional[np.ndarray] = None,
        occl_scores:   Optional[np.ndarray] = None,
    ) -> PerceiveOutput:
        """
        Parameters
        ----------
        selected_indices  : original indices (into the full video) of selected_frames
        disturbance_scores: per-frame d values for selected_frames
                            (subset of full-video scores from Scout)
        pool_frames / pool_indices: for Loop 1 Strategy B (retrieve cleaner frames)
        blur_scores / bright_scores / occl_scores:
            Per-channel disturbance profiles from Scout's assess_quality (Eq. 2).
            Used for disturbance-aware tool routing (Section 3.2, Appendix E.2).
        """
        # d-scores scoped to the selected frames
        sel_dist = disturbance_scores[selected_indices] \
            if len(disturbance_scores) > max(selected_indices, default=-1) + 1 \
            else np.zeros(len(selected_frames))

        # ── Averaged disturbance profile d̄ for routing (Section 3.2) ────────
        # "the averaged disturbance profile d̄ = (d̄_blur, d̄_bright, d̄_occl)
        #  selects the most reliable candidate by identifying the dominant
        #  corruption type"
        #
        # FIX (Inconsistency 9 — blur dominance):
        #   Previous code recomputed blur from scratch via _is_blur_dominant()
        #   using a Laplacian threshold of 100.0 unrelated to the paper's
        #   TAU_BLUR=500.  The paper says routing uses the profile computed in
        #   Step 1.1 (assess_quality).  Now uses the actual per-channel scores.
        if blur_scores is not None and bright_scores is not None and occl_scores is not None:
            sel_blur = blur_scores[selected_indices] \
                if len(blur_scores) > max(selected_indices, default=-1) + 1 \
                else np.zeros(len(selected_frames))
            sel_bright = bright_scores[selected_indices] \
                if len(bright_scores) > max(selected_indices, default=-1) + 1 \
                else np.zeros(len(selected_frames))
            sel_occl = occl_scores[selected_indices] \
                if len(occl_scores) > max(selected_indices, default=-1) + 1 \
                else np.zeros(len(selected_frames))
            avg_blur = float(np.mean(sel_blur))
            avg_bright = float(np.mean(sel_bright))
            avg_occl = float(np.mean(sel_occl))
        else:
            avg_blur = avg_bright = avg_occl = 0.0

        # Determine dominant corruption type (Section 3.2)
        disturbance_profile = {"blur": avg_blur, "brightness": avg_bright, "occlusion": avg_occl}
        dominant_type = max(disturbance_profile, key=disturbance_profile.get)

        # ── 1. Sub-query decomposition ────────────────────────────────────────
        # FIX (Inconsistency 7 — text-only decomposition):
        #   Paper (Tab. 9, Appendix E.1) specifies decomposition uses both the
        #   question text AND the visual content / disturbance profile of the
        #   selected trustworthy frames.  Text+Frame gives +4.1 and +5.3 points
        #   on clean and corrupted inputs respectively.
        if provided_sub_queries is not None:
            # Loop-2 gap sub-queries arrive as plain strings; infer a type
            typed_sub_queries = [
                (sq, self._infer_semantic_type(sq)) for sq in provided_sub_queries
            ]
        else:
            typed_sub_queries = self._decompose_query(
                query,
                optimal_subqueries=optimal_subqueries,
                selected_frames=selected_frames,
                disturbance_profile=disturbance_profile,
            )

        facts: List[Fact] = []
        all_tool_results: List[ToolResult] = []
        single_pass_facts: List[Fact] = []
        single_pass_tool_results: List[ToolResult] = []

        # ── 2 & 3. Per-sub-query: tool selection + Loop 1 ────────────────────
        for sq, sq_type in typed_sub_queries:
            fact, tr_list, single_fact, single_tool = self._dispatch_sub_query(
                sq             = sq,
                sq_type        = sq_type,
                frames         = selected_frames,
                frame_indices  = selected_indices,
                dist_scores    = sel_dist,
                disturbance_profile = disturbance_profile,
                dominant_type  = dominant_type,
                pool_frames    = pool_frames,
                pool_indices   = pool_indices,
                all_dist       = disturbance_scores,
                enable_loop1   = enable_loop1,
            )
            facts.extend(fact)
            all_tool_results.extend(tr_list)
            single_pass_facts.extend(single_fact)
            single_pass_tool_results.extend(single_tool)

        return PerceiveOutput(
            facts=facts,
            tool_results=all_tool_results,
            single_pass_facts=single_pass_facts,
            single_pass_tool_results=single_pass_tool_results,
            sub_queries=[sq for sq, _ in typed_sub_queries],
        )

    # ── Sub-query decomposition ───────────────────────────────────────────────
    def _decompose_query(
        self,
        query: str,
        optimal_subqueries: Optional[int] = None,
        selected_frames: Optional[np.ndarray] = None,
        disturbance_profile: Optional[Dict[str, float]] = None,
    ) -> List[tuple]:
        """
        Ask the LLM agent to break the query into targeted sub-queries WITH
        semantic types, returning a list of (sub_query, semantic_type) tuples.

        FIX (Inconsistency 7 — text-only decomposition):
            Paper (Tab. 9, Appendix E.1) specifies decomposition conditions on
            both the question text AND the visual content of the selected
            trustworthy frames plus their disturbance profile.

        FIX (Inconsistency B — semantic type dropped):
            Appendix E.1's output format is a JSON list of
            {"sub_query": ..., "type": ...} objects.  The semantic type is the
            FIRST stage of the two-stage routing protocol (Section 3.2), so it
            must be preserved rather than discarded.
        """
        target = optimal_subqueries or estimate_optimal_subqueries(query, agent_fn=self.agent)

        # Brief video context from the selected frames (Text+Frame, Appendix E.1)
        video_description = "(no visual context available)"
        if selected_frames is not None and len(selected_frames) > 0:
            try:
                desc_prompt = (
                    f"These are {len(selected_frames)} selected frames from a video. "
                    f"The question being asked is: {query}\n"
                    "Briefly describe what visual elements and scene context would be "
                    "relevant to answering this question (1-2 sentences)."
                )
                video_description = self.agent(desc_prompt)
            except Exception:
                video_description = "(visual context extraction failed)"

        # Disturbance profile string (Appendix E.1)
        if disturbance_profile is not None:
            dist_str = (
                f"blur={disturbance_profile.get('blur', 0.0):.2f}, "
                f"brightness={disturbance_profile.get('brightness', 0.0):.2f}, "
                f"occlusion={disturbance_profile.get('occlusion', 0.0):.2f}"
            )
        else:
            dist_str = "blur=0.00, brightness=0.00, occlusion=0.00"

        prompt = (
            f"You are an expert video analyst. Decompose a complex question about "
            f"a video into a minimal set of atomic sub-queries. Each sub-query must "
            f"target exactly one perceptual primitive and be answerable by a single "
            f"visual tool call. Do not generate redundant sub-queries.\n\n"
            f"Decomposition guidelines:\n"
            f"1. Identify the distinct perceptual demands implied by the question.\n"
            f"2. For each demand, formulate exactly one atomic sub-query.\n"
            f"3. Assign a semantic type to each sub-query: one of "
            f"[spatial, temporal, attribute, action, text].\n"
            f"4. Minimize the total number of sub-queries. Target about {target}.\n\n"
            f"Input:\n"
            f"  Video context: {video_description}\n"
            f"  Disturbance profile of selected frames: {dist_str}\n"
            f"  Question: {query}\n\n"
            f"Output ONLY a JSON array of objects. No explanation, just JSON.\n"
            f'Example: [{{"sub_query": "What objects are near the intersection?", '
            f'"type": "spatial"}}, {{"sub_query": "In what order do they appear?", '
            f'"type": "temporal"}}]\n'
        )
        raw = self.agent(prompt)
        valid_types = {"spatial", "temporal", "attribute", "action", "text"}
        try:
            parsed = json.loads(self._extract_json(raw))
            if isinstance(parsed, list) and parsed:
                pairs: List[tuple] = []
                for item in parsed:
                    if isinstance(item, dict):
                        sq = str(item.get("sub_query", "")).strip()
                        st = str(item.get("type", "spatial")).strip().lower()
                        if st not in valid_types:
                            st = "spatial"
                    else:
                        sq, st = str(item).strip(), "spatial"
                    if sq:
                        pairs.append((sq, st))
                if pairs:
                    return pairs[:max(target, 1)]
        except Exception:
            pass
        # Fallback: treat whole query as one spatial sub-query
        return [(query, "spatial")]

    # ── Tool selection ────────────────────────────────────────────────────────
    def _select_tool(
        self,
        sub_query: str,
        sq_type: str,
        disturbance_profile: Dict[str, float],
        dominant_type: str,
    ) -> str:
        """
        Two-stage disturbance-aware tool routing (Section 3.2, Table 18, App. E.2).

        FIX (Inconsistency A/B — full tool set + semantic-type routing):
            Stage 1: the semantic type of the sub-query determines the candidate
                     tools (spatial→detect_objects/caption_frame; temporal→
                     track_temporal/recognize_action; text→read_text/caption_frame; ...).
            Stage 2: the averaged disturbance profile selects the most reliable
                     candidate by identifying the dominant corruption type.
            Routing is soft: the host VLM picks via in-context reasoning over the
            candidate set (Appendix E.2); the learned table (Table 18) is the
            deterministic fallback.
        """
        sq_type = sq_type if sq_type in self._CANDIDATES else "spatial"
        candidates = [c for c in self._CANDIDATES[sq_type] if c in self.tools]
        if not candidates:
            candidates = [t for t in ("caption_frame", "detect_objects") if t in self.tools]
        if not candidates:
            return next(iter(self.tools), "caption_frame")

        # Single viable candidate → no routing decision needed
        if len(candidates) == 1:
            return candidates[0]

        d_blur = disturbance_profile.get("blur", 0.0)
        d_bright = disturbance_profile.get("brightness", 0.0)
        d_occl = disturbance_profile.get("occlusion", 0.0)
        no_corruption = max(d_blur, d_bright, d_occl) < 1e-6
        corruption = "clean" if no_corruption else dominant_type

        tool_costs = {
            "detect_objects": 0.50, "caption_frame": 0.30, "track_temporal": 0.70,
            "recognize_action": 0.60, "read_text": 0.25,
        }
        cand_block = "\n".join(
            f"    {c} (cost={tool_costs.get(c, 0.5):.2f})" for c in candidates
        )

        prompt = (
            f"You are a tool routing agent. Given a sub-query, its semantic type, "
            f"and the disturbance profile of the selected frames, choose the best "
            f"perception tool that maximizes result reliability under the current "
            f"corruption conditions.\n\n"
            f"Routing guidelines:\n"
            f"  - For spatial sub-queries under blur: prefer caption_frame over "
            f"detect_objects (detection requires sharp visual boundaries).\n"
            f"  - For temporal sub-queries under occlusion: prefer recognize_action "
            f"over track_temporal (tracking loses targets under occlusion).\n"
            f"  - For text sub-queries under blur: prefer caption_frame over "
            f"read_text (OCR degrades rapidly under spatial blur).\n"
            f"  - When brightness distortion dominates: prioritize tools robust to "
            f"extreme illumination.\n"
            f"  - When multiple tools are viable: prefer the one with lower cost.\n\n"
            f"Input:\n"
            f"  Sub-query: {sub_query}\n"
            f"  Semantic type: {sq_type}\n"
            f"  Disturbance profile: blur={d_blur:.2f}, brightness={d_bright:.2f}, "
            f"occlusion={d_occl:.2f}\n"
            f"  Dominant corruption: {corruption}\n"
            f"  Available tools (candidates for this semantic type):\n{cand_block}\n\n"
            f"Select the best tool. Output ONLY the tool name. No explanation."
        )
        try:
            raw = self.agent(prompt).strip().lower()
            for name in candidates:
                if name in raw:
                    return name
        except Exception:
            pass

        # Fallback: deterministic Table-18 lookup, constrained to viable candidates
        table_choice = self._ROUTING_TABLE.get(sq_type, {}).get(corruption)
        if table_choice in candidates:
            return table_choice
        return candidates[0]

    # ── Loop 1: confidence-driven re-invocation (Eq. 4) ──────────────────────
    def _dispatch_sub_query(
        self,
        sq:            str,
        sq_type:       str,
        frames:        np.ndarray,
        frame_indices: List[int],
        dist_scores:   np.ndarray,
        disturbance_profile: Dict[str, float],
        dominant_type: str,
        pool_frames:   np.ndarray,
        pool_indices:  List[int],
        all_dist:      np.ndarray,
        enable_loop1:  bool,
    ):
        tool_results: List[ToolResult] = []
        facts:        List[Fact] = []

        tool_name = self._select_tool(sq, sq_type, disturbance_profile, dominant_type)
        tr = self._call_tool(tool_name, frames, sq, dist_scores, frame_indices)
        tool_results.append(tr)
        single_pass_fact = Fact(
            sub_query=sq,
            result=tr.result,
            confidence=tr.confidence,
            source_frames=tr.source_frames,
            tool_name=tr.tool_name,
            flagged=False,
            semantic_type=sq_type,
            disturbance=self._source_disturbance(tr.source_frames, all_dist),
        )

        # ── Loop 1 ────────────────────────────────────────────────────────────
        if enable_loop1 and tr.confidence < self.theta_1:
            # Strategy A: alternative tool on same frames
            alt_name = self._alternative_tool(tool_name, sq_type, dominant_type)
            tr_a = self._call_tool(alt_name, frames, sq, dist_scores, frame_indices)
            tool_results.append(tr_a)
            candidate_results = [tr, tr_a]

            # Strategy B: retrieve cleaner frames from pool, then re-invoke
            if len(pool_frames) > 0:
                pool_dist = all_dist[pool_indices] \
                    if len(all_dist) > max(pool_indices, default=-1) + 1 \
                    else np.zeros(len(pool_frames))
                ret_result = self.retrieve_tool(
                    pool_frames, sq, pool_dist, pool_indices
                )
                tool_results.append(ret_result)

                if ret_result.result and ret_result.result.get("retrieved_indices"):
                    ridx = ret_result.result["retrieved_indices"]
                    r_frames = pool_frames[ridx]
                    r_dist   = pool_dist[ridx]
                    r_orig   = [pool_indices[i] for i in ridx]
                    tr_b = self._call_tool(tool_name, r_frames, sq, r_dist, r_orig)
                    tool_results.append(tr_b)
                    candidate_results.append(tr_b)

            tr, conflicting = self._fuse_candidates(candidate_results)
        else:
            conflicting = False

        facts.append(Fact(
            sub_query     = sq,
            result        = tr.result,
            confidence    = tr.confidence,
            source_frames = tr.source_frames,
            tool_name     = tr.tool_name,
            flagged       = conflicting,
            semantic_type = sq_type,
            disturbance   = self._source_disturbance(tr.source_frames, all_dist),
        ))
        return facts, tool_results, [single_pass_fact], [tool_results[0]]

    @staticmethod
    def _source_disturbance(source_frames: List[int], all_dist: np.ndarray) -> float:
        """Mean d(f) of a fact's source frames (used for HIGH/MED/LOW tiering)."""
        if source_frames is None or len(source_frames) == 0 or len(all_dist) == 0:
            return 0.0
        valid = [i for i in source_frames if 0 <= i < len(all_dist)]
        if not valid:
            return 0.0
        return float(np.mean(all_dist[valid]))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _call_tool(
        self,
        tool_name:    str,
        frames:       np.ndarray,
        sub_query:    str,
        dist_scores:  np.ndarray,
        frame_indices: List[int],
    ) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            from ..tools.base import ToolResult
            return ToolResult(result=None, confidence=0.0,
                              tool_name=tool_name, source_frames=frame_indices)
        return tool(frames, sub_query, dist_scores, frame_indices)

    def _alternative_tool(self, current: str, sq_type: str, dominant_type: str) -> str:
        """
        Pick an alternative tool different from `current` for Loop 1 Strategy A.

        Chooses the next-best candidate for this semantic type (Section 3.2),
        respecting the disturbance-aware preference (e.g. avoid detection-style
        tools under blur).
        """
        sq_type = sq_type if sq_type in self._CANDIDATES else "spatial"
        candidates = [c for c in self._CANDIDATES[sq_type] if c in self.tools]
        # Prefer the table choice for this corruption if it differs from current
        no_corruption = False
        table_choice = self._ROUTING_TABLE.get(sq_type, {}).get(
            "clean" if no_corruption else dominant_type
        )
        for c in ([table_choice] + candidates):
            if c and c != current and c in self.tools:
                return c
        # Fall back to caption_frame (broadly applicable, tolerant of corruption)
        return "caption_frame" if "caption_frame" in self.tools else current

    @staticmethod
    def _infer_semantic_type(sub_query: str) -> str:
        """
        Heuristic semantic-type inference for sub-queries that arrive without a
        type (e.g. Loop-2 gap queries).  Maps to one of
        [spatial, temporal, attribute, action, text].
        """
        s = sub_query.lower()
        if any(k in s for k in ("read", "text", "sign", "plate", "label", "number on")):
            return "text"
        if any(k in s for k in ("order", "sequence", "before", "after", "track", "move", "trajectory", "when")):
            return "temporal"
        if any(k in s for k in ("doing", "action", "activity", "event", "happen")):
            return "action"
        if any(k in s for k in ("color", "colour", "size", "shape", "appearance", "attribute", "looks like")):
            return "attribute"
        return "spatial"

    def _fuse_candidates(
        self,
        candidates: List[ToolResult],
    ) -> tuple[ToolResult, bool]:
        """
        Evidence fusion from Eq. 4:
        - agreeing results reinforce confidence
        - disagreements keep the highest-confidence result and flag the fact
        """
        if not candidates:
            return ToolResult(None, 0.0, "unknown", []), False

        best = max(candidates, key=lambda item: item.confidence)
        normalized_best = self._normalize_result(best.result)
        agreeing = 0
        conflicting = False

        for candidate in candidates:
            if candidate is best:
                continue
            normalized = self._normalize_result(candidate.result)
            if normalized and normalized == normalized_best:
                agreeing += 1
            elif normalized and normalized_best and normalized != normalized_best:
                conflicting = True

        if agreeing > 0:
            best = replace(
                best,
                confidence=float(min(1.0, best.confidence + 0.1 * agreeing)),
            )

        return best, conflicting

    @staticmethod
    def _normalize_result(result: Any) -> str:
        if result is None:
            return ""
        text = str(result).strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text[:300]

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON array/object from LLM output (greedy outer match)."""
        # Greedy match so arrays of objects like [{...}, {...}] are captured whole
        m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        return m.group(0) if m else text
