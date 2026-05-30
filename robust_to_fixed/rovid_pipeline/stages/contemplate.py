"""
RoVid — Stage 2: Contemplate — Reasoning with Active Inquiry  (Section 0.4)

The paper defines exactly TWO stages:

    Stage 1 — Tool-Driven Visual Gathering   (Steps 1.1 – 1.4, see stage1.py)
    Stage 2 — Contemplate                    (this module)

Stage 1 produces evidence; Contemplate decides what that evidence means.  The
critical distinction is that Contemplate does not passively consume facts — it
actively evaluates their reliability, identifies what is missing, and, when
necessary, drives the pipeline back into Stage 1 for additional tool invocations.

Three mechanisms (Section 0.4):

    Confidence-Weighted Reasoning
        Surfaces cj as a first-class reasoning signal in the prompt so that the
        LLM discounts low-confidence (degraded-frame) observations.

    Loop 2: Gap-Driven Active Inquiry
        When evidence is insufficient the LLM generates new sub-queries and
        dispatches them back through Stage 1's tool-calling pipeline (Steps 1.3–
        1.4, including Loop 1).  The resulting facts are fused and reasoning
        restarts.

    Loop 3: Uncertainty-Triggered Look-Back  (Eq. 5)
        If overall reasoning entropy still exceeds θ₂ after Loop 2, the agent
        revisits the retrieval pool P assembled during Stage 1.  The refined
        query q' = q ⊕ Stage 1 output (enriched by partial reasoning) enables
        more targeted retrieval.  Loop 3 executes at most twice.

        Entropy is computed by the formula-based _reasoning_entropy() method
        (mean low-confidence contribution + flagged-fact penalty) rather than by
        LLM self-report, giving a deterministic, calibrated trigger.

Three-Loop Hierarchy (Section 0.4):
    Step 1.4 (tool-level) ⊂ Loop 2 (reasoning-level) ⊂ Loop 3 (retrieval-level)
    Cheap local corrections are attempted before expensive global ones.

Training mode (Section 0.5):
    Pass training=True to disable Loop 2 and Loop 3, producing a single-pass
    rollout that keeps reward computation clean and variance-free.
"""

from __future__ import annotations
import math
import re
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..tools.base import ToolBase, ToolResult
from ..tools.selection_tools import RetrieveFrames
from .perceive import Fact, Perceive, PerceiveOutput


# ── Thresholds from paper ─────────────────────────────────────────────────────
THETA_2             = 0.6   # reasoning entropy threshold that triggers Loop 3
MAX_LOOP3_ITERS     = 2     # Loop 3 executes at most twice (paper: "to bound inference cost")
MAX_LOOP2_ITERS     = 3     # practical bound on Gap-Driven inquiry


@dataclass
class ContemplateOutput:
    answer:       str
    reasoning:    str
    final_facts:  List[Fact]
    loop2_iters:  int = 0
    loop3_iters:  int = 0
    tool_results: List[ToolResult] = field(default_factory=list)


@dataclass
class ReasoningPass:
    reasoning: str
    answer: str
    gaps: List[str]
    uncertainty: float
    raw_response: str


class Contemplate:
    """
    Stage 2: Contemplate — Reasoning with Active Inquiry  (Section 0.4)

    Parameters
    ----------
    agent_fn         : fn(prompt: str) -> str   (LLM text-only, no vision)
    perceive_stage   : Perceive instance (re-used for Loop 2 sub-queries & Loop 3
                       re-perception; this drives Stage 1 Steps 1.3–1.4 including
                       Loop 1)
    retrieve_tool    : RetrieveFrames tool for Loop 3  (Eq. 5)
    theta_2          : reasoning-entropy threshold for Loop 3 trigger
    """

    def __init__(
        self,
        agent_fn:      Callable[[str], str],
        perceive_stage: Perceive,
        retrieve_tool: RetrieveFrames,
        theta_2:        float = THETA_2,
    ):
        self.agent    = agent_fn
        self.perceive = perceive_stage
        self.retrieve = retrieve_tool
        self.theta_2  = theta_2

    # ── Public entry-point ────────────────────────────────────────────────────
    def run(
        self,
        query:              str,
        facts:              List[Fact],
        pool_frames:        np.ndarray,
        pool_indices:       List[int],
        disturbance_scores: np.ndarray,
        selected_frames:    np.ndarray,
        selected_indices:   List[int],
        training:           bool = False,
    ) -> ContemplateOutput:
        """
        Execute Stage 2: Contemplate — Reasoning with Active Inquiry.

        Parameters
        ----------
        training : when True, Loops 2 and 3 are skipped entirely and the stage
                   produces a single reasoning pass.  This enforces the GRPO
                   training constraint from Section 0.5: "no gap-driven active
                   inquiry (Loop 2), and no uncertainty-triggered look-back
                   (Loop 3)".  Only the correctness, sub-query efficiency, and
                   confidence-cost reward components are well-defined under this
                   single-pass constraint.
        """
        all_tool_results: List[ToolResult] = []
        loop2_iters = 0
        loop3_iters = 0
        current_pass = self.reason_once(query, facts)

        # ── Training mode: single forward pass only (Section 0.5) ─────────────
        # During GRPO training the rollout executes Steps 1.1–1.3 in a single
        # forward pass.  Contemplate must also run only once so reward computation
        # operates on a fixed-length trajectory with no loop-induced variance.
        if training:
            return ContemplateOutput(
                answer       = self._extract_answer(current_pass.answer),
                reasoning    = current_pass.reasoning,
                final_facts  = facts,
                loop2_iters  = 0,
                loop3_iters  = 0,
                tool_results = [],
            )

        # ── Confidence-Weighted Reasoning + Loop 2: Gap-Driven Active Inquiry ──
        for _ in range(MAX_LOOP2_ITERS):
            if not current_pass.gaps:
                break   # no information gaps → done

            loop2_iters += 1
            # Dispatch gap sub-queries back through Stage 1 Steps 1.3–1.4
            # (enable_loop1=True so Step 1.4 is active, as per Section 0.4)
            new_perceive: PerceiveOutput = self.perceive.run(
                query              = query,
                selected_frames    = selected_frames,
                selected_indices   = selected_indices,
                disturbance_scores = disturbance_scores,
                pool_frames        = pool_frames,
                pool_indices       = pool_indices,
                enable_loop1       = True,
                provided_sub_queries = current_pass.gaps,
            )
            all_tool_results.extend(new_perceive.tool_results)
            facts = self._fuse_facts(facts, new_perceive.facts)
            current_pass = self.reason_once(query, facts)

        # ── Loop 3: Uncertainty-Triggered Look-Back ───────────────────────────
        # The trigger is the formula-based _reasoning_entropy() (Section 0.4):
        #   entropy = mean(1 - c_j) + flagged_penalty
        # This is deterministic and calibrated via the same confidence signals
        # that drive the rest of the pipeline.  LLM self-reported uncertainty
        # (the UNCERTAINTY: field in the reasoning prompt) is informational only
        # and is NOT used as the Loop 3 trigger.
        for _ in range(MAX_LOOP3_ITERS):
            entropy = self._reasoning_entropy(facts)
            if entropy <= self.theta_2:
                break

            loop3_iters += 1

            # Eq. 5: q' = q ⊕ Stage 1 output
            # "Stage 1 output" is the fact set F produced by Stage 1 (Steps 1.3–
            # 1.4) that seeded this reasoning pass, now enriched by the partial
            # reasoning performed so far, making the refined query more specific
            # than the original and enabling more targeted retrieval.
            partial_summary = self._summarise_facts(facts)
            refined_query   = f"{query} | Context from analysis: {partial_summary}"

            if len(pool_frames) == 0:
                break

            pool_dist = disturbance_scores[pool_indices] \
                if len(disturbance_scores) > max(pool_indices, default=-1) + 1 \
                else np.zeros(len(pool_frames))

            # retrieve_frames(P, q', {d(f)})
            ret_result = self.retrieve(
                pool_frames, refined_query, pool_dist, pool_indices
            )
            all_tool_results.append(ret_result)

            if ret_result.result is None or not ret_result.result.get("retrieved_indices"):
                break

            ridx     = ret_result.result["retrieved_indices"]
            r_frames = pool_frames[ridx]
            r_dist   = pool_dist[ridx]
            r_orig   = [pool_indices[i] for i in ridx]

            # Re-perceive retrieved frames (Step 1.4 / Loop 1 active)
            re_perceive: PerceiveOutput = self.perceive.run(
                query             = refined_query,
                selected_frames   = r_frames,
                selected_indices  = r_orig,
                disturbance_scores= disturbance_scores,
                pool_frames       = np.empty((0, *pool_frames.shape[1:]), dtype=pool_frames.dtype),
                pool_indices      = [],
                enable_loop1      = True,
            )
            all_tool_results.extend(re_perceive.tool_results)
            facts = self._fuse_facts(facts, re_perceive.facts)
            current_pass = self.reason_once(query, facts)

        return ContemplateOutput(
            answer       = self._extract_answer(current_pass.answer),
            reasoning    = current_pass.reasoning,
            final_facts  = facts,
            loop2_iters  = loop2_iters,
            loop3_iters  = loop3_iters,
            tool_results = all_tool_results,
        )

    def reason_once(self, query: str, facts: List[Fact]) -> ReasoningPass:
        return self._reason_with_confidence(query, facts)

    # ── Confidence-Weighted Reasoning ─────────────────────────────────────────
    def _reason_with_confidence(
        self, query: str, facts: List[Fact]
    ) -> ReasoningPass:
        """
        Confidence-Weighted Evidence Synthesis (Section 3.1, Appendix E.3).

        Groups evidence into three reliability tiers (HIGH/MEDIUM/LOW) based
        on confidence and disturbance thresholds, then synthesises the answer
        primarily from HIGH-tier evidence.

        FIX (Inconsistency 6 — synthesis prompt):
            Previous code used a generic "weight reliable evidence more heavily"
            prompt without implementing the explicit three-tier grouping described
            in the paper (Section 3.1, Appendix E.3).  The paper specifies:
                HIGH:   c_j >= 0.7 and d < 0.3
                MEDIUM: all other evidence
                LOW:    c_j < 0.3 or d >= 0.7
            with specific rules for how each tier contributes to the answer.
        """
        # ── Tier assignment (Section 3.1, Appendix E.3) ──────────────────────
        # HIGH:   c_j >= 0.7 AND d < 0.3
        # LOW:    c_j <  0.3  OR d >= 0.7
        # MEDIUM: everything else
        #
        # FIX (Inconsistency C — tier assignment used a confidence proxy):
        #   The tier rules require BOTH the confidence c_j AND the disturbance d
        #   of the fact's source frames.  Facts now carry the actual mean
        #   source-frame disturbance (Fact.disturbance), so the thresholds match
        #   the paper exactly instead of approximating d from confidence.
        high_facts, med_facts, low_facts = [], [], []
        for f in facts:
            d = float(getattr(f, "disturbance", 0.0))
            if f.confidence >= 0.7 and d < 0.3:
                high_facts.append(f)
            elif f.confidence < 0.3 or d >= 0.7:
                low_facts.append(f)
            else:
                med_facts.append(f)

        def _format_facts(fact_list: List[Fact], tier: str) -> str:
            if not fact_list:
                return f"  (no {tier}-tier evidence)\n"
            lines = []
            for i, f in enumerate(fact_list):
                flag_note = " [FLAGGED: possible contradiction]" if f.flagged else ""
                result_str = self._result_to_str(f.result)
                lines.append(
                    f"  [{tier} {i+1}] tool={f.tool_name} | conf={f.confidence:.2f}{flag_note}\n"
                    f"    sub-query: {f.sub_query}\n"
                    f"    evidence:  {result_str}"
                )
            return "\n".join(lines) + "\n"

        facts_block = (
            "HIGH-tier evidence (confidence >= 0.7, low disturbance):\n"
            + _format_facts(high_facts, "HIGH")
            + "\nMEDIUM-tier evidence:\n"
            + _format_facts(med_facts, "MED")
            + "\nLOW-tier evidence (confidence < 0.3 or high disturbance):\n"
            + _format_facts(low_facts, "LOW")
        )

        prompt = (
            f"Question: {query}\n\n"
            "Evidence grouped by reliability tier:\n"
            f"{facts_block}\n"
            "Synthesis rules:\n"
            "1. Build your answer primarily from HIGH-tier evidence.\n"
            "2. Use MEDIUM-tier evidence only if it is consistent with HIGH-tier "
            "conclusions; discard it if contradictory.\n"
            "3. Use LOW-tier evidence only when no HIGH-tier evidence exists, "
            "and explicitly note the uncertainty.\n"
            "4. If all evidence is LOW-tier, state that the answer is uncertain.\n\n"
            "Format your response as:\n"
            "REASONING: <step-by-step reasoning considering evidence reliability "
            "and tier-based synthesis>\n"
            "GAPS: <comma-separated list of missing info, or NONE>\n"
            "UNCERTAINTY: <number between 0 and 1>\n"
            "ANSWER: <your best answer to the question>"
        )

        raw = self.agent(prompt)

        reasoning = self._extract_section(raw, "REASONING")
        gaps_str  = self._extract_section(raw, "GAPS")
        uncertainty_str = self._extract_section(raw, "UNCERTAINTY")
        answer    = self._extract_section(raw, "ANSWER")

        gaps: List[str] = []
        if gaps_str and gaps_str.upper() != "NONE":
            gaps = [g.strip() for g in gaps_str.split(",") if g.strip()]
        uncertainty = self._parse_uncertainty(uncertainty_str, facts)

        return ReasoningPass(
            reasoning=reasoning,
            answer=answer,
            gaps=gaps,
            uncertainty=uncertainty,
            raw_response=raw,
        )

    # ── Loop 3 helpers ────────────────────────────────────────────────────────
    def _reasoning_entropy(self, facts: List[Fact]) -> float:
        """
        Proxy for reasoning uncertainty:
        higher when facts are low-confidence or flagged.
        """
        if not facts:
            return 1.0
        # Mean low-confidence contribution + flagged penalty
        low_conf_signal = np.mean([1.0 - f.confidence for f in facts])
        flag_penalty    = np.mean([0.3 if f.flagged else 0.0 for f in facts])
        entropy = float(np.clip(low_conf_signal + flag_penalty, 0.0, 1.0))
        return entropy

    def _parse_uncertainty(self, text: str, facts: List[Fact]) -> float:
        match = re.search(r"([01](?:\.\d+)?)", text)
        if match:
            value = float(match.group(1))
            return float(np.clip(value, 0.0, 1.0))
        return self._reasoning_entropy(facts)

    def _summarise_facts(self, facts: List[Fact]) -> str:
        """Produce a short textual summary of current facts for query enrichment."""
        parts = []
        for f in sorted(facts, key=lambda x: x.confidence, reverse=True)[:5]:
            parts.append(f"{f.sub_query}: {self._result_to_str(f.result)[:80]}")
        return "; ".join(parts)

    # ── Evidence fusion ───────────────────────────────────────────────────────
    def _fuse_facts(self, old: List[Fact], new: List[Fact]) -> List[Fact]:
        """
        Consistent results mutually reinforce confidence.
        Contradictions preserve the higher-confidence version and flag it.
        (Section 1.4, evidence fusion note)
        """
        merged: Dict[str, Fact] = {f.sub_query: f for f in old}
        for nf in new:
            if nf.sub_query in merged:
                existing = merged[nf.sub_query]
                if self._normalize_result(nf.result) == self._normalize_result(existing.result):
                    merged[nf.sub_query] = replace(
                        existing,
                        confidence=float(min(1.0, max(existing.confidence, nf.confidence) + 0.1)),
                        flagged=existing.flagged or nf.flagged,
                    )
                else:
                    winner = nf if nf.confidence > existing.confidence else existing
                    merged[nf.sub_query] = replace(winner, flagged=True)
            else:
                merged[nf.sub_query] = nf
        return list(merged.values())

    # ── Utility ───────────────────────────────────────────────────────────────
    @staticmethod
    def _result_to_str(result: Any) -> str:
        if result is None:
            return "(empty)"
        if isinstance(result, str):
            return result[:200]
        if isinstance(result, list):
            return " | ".join(str(r)[:60] for r in result[:3])
        if isinstance(result, dict):
            return str({k: str(v)[:40] for k, v in list(result.items())[:3]})
        return str(result)[:200]

    @staticmethod
    def _extract_section(text: str, tag: str) -> str:
        pattern = rf"{tag}:\s*(.*?)(?=\n[A-Z]+:|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Pull out letter answer (A/B/C/D) or free-form answer."""
        # Multiple-choice: look for standalone letter
        m = re.search(r"\b([A-D])\b", text)
        return m.group(1) if m else text.strip()

    @staticmethod
    def _normalize_result(result: Any) -> str:
        if result is None:
            return ""
        text = str(result).strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text[:300]
