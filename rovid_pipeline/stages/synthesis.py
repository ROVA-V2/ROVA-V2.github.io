"""
Robust-TO — Confidence-Weighted Evidence Synthesis  (Section 3.1, App. E.3)

The paper is explicit that synthesis is a SINGLE reasoning pass:

    "Once all sub-queries have been answered, the host VLM consolidates the
     accumulated evidence F and produces a final answer in a single reasoning
     pass."  (Section 3.1)

    "When the input video is clean, every fact belongs to the high-reliability
     tier, the grouping step becomes trivial, and the procedure reduces to
     ordinary multi-fact reasoning with no overhead."  (Section 3.1)

There are NO inference-time loops (no gap-driven inquiry, no uncertainty look-back).
Synthesis groups facts into three reliability tiers and reasons over them once:

    HIGH:   c_j >= 0.7  AND  d < 0.3          (clean frames, confident tools)
    LOW:    c_j <  0.3  OR   d >= 0.7          (degraded frames or low confidence)
    MEDIUM: everything else

    1. Build the conclusion primarily from HIGH-tier facts.
    2. Use MEDIUM-tier facts only if consistent with HIGH-tier; discard if not.
    3. Use LOW-tier facts only when no HIGH-tier evidence exists, and note the
       uncertainty.

The thresholds match the synthesis prompt in App. E.3 / Table 16, and the output
format is the paper's <think>...</think><answer>X</answer> contract.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Callable, List

from .perception import Fact


# Tier thresholds (App. E.3 / Table 16)
HIGH_CONF, HIGH_DIST = 0.7, 0.3
LOW_CONF,  LOW_DIST  = 0.3, 0.7


@dataclass
class SynthesisOutput:
    answer:    str
    reasoning: str
    raw:       str


class ConfidenceWeightedSynthesis:
    """Single-pass three-tier evidence synthesis (Section 3.1)."""

    def __init__(self, agent_fn: Callable[[str], str]):
        self.agent = agent_fn

    def run(self, query: str, facts: List[Fact]) -> SynthesisOutput:
        high, med, low = self._tier(facts)
        block = (
            "HIGH-tier evidence (confidence >= 0.7 AND low disturbance):\n"
            + self._fmt(high, "HIGH")
            + "\nMEDIUM-tier evidence:\n"
            + self._fmt(med, "MED")
            + "\nLOW-tier evidence (confidence < 0.3 OR high disturbance):\n"
            + self._fmt(low, "LOW")
        )
        prompt = (
            "[Task] You are synthesizing evidence collected from multiple visual "
            "tools to answer a question about a video. Each piece of evidence has "
            "a confidence score and a source-frame disturbance level. Produce a "
            "reliable answer grounded in the most trustworthy evidence.\n\n"
            f"[Question] {query}\n\n"
            f"[Evidence grouped by reliability tier]\n{block}\n"
            "[Synthesis rules]\n"
            "1. Build your answer primarily from HIGH-tier evidence.\n"
            "2. Use MEDIUM-tier evidence only if consistent with HIGH-tier "
            "conclusions; discard it if contradictory.\n"
            "3. Use LOW-tier evidence only when no HIGH-tier evidence exists, and "
            "explicitly note the uncertainty.\n"
            "4. If all evidence is LOW-tier, state that the answer is uncertain.\n\n"
            "[Output format] Provide step-by-step reasoning inside <think> tags, "
            "then the final answer inside <answer> tags. Output only these two blocks.\n"
            "<think> ... </think>\n<answer>X</answer>"
        )
        raw = self.agent(prompt)
        reasoning = self._extract(raw, "think")
        answer    = self._extract_answer(self._extract(raw, "answer") or raw)
        return SynthesisOutput(answer=answer, reasoning=reasoning, raw=raw)

    # ── Tier assignment (App. E.3) ───────────────────────────────────────────
    @staticmethod
    def _tier(facts: List[Fact]):
        high, med, low = [], [], []
        for f in facts:
            d = float(getattr(f, "disturbance", 0.0))
            if f.confidence >= HIGH_CONF and d < HIGH_DIST:
                high.append(f)
            elif f.confidence < LOW_CONF or d >= LOW_DIST:
                low.append(f)
            else:
                med.append(f)
        return high, med, low

    @staticmethod
    def _fmt(facts: List[Fact], tier: str) -> str:
        if not facts:
            return f"  (no {tier}-tier evidence)\n"
        lines = []
        for i, f in enumerate(facts):
            flag = " [FLAGGED: possible contradiction]" if f.flagged else ""
            lines.append(
                f"  [{tier} {i+1}] tool={f.tool_name} | conf={f.confidence:.2f}{flag}\n"
                f"    sub-query: {f.sub_query}\n"
                f"    evidence:  {ConfidenceWeightedSynthesis._result_to_str(f.result)}"
            )
        return "\n".join(lines) + "\n"

    # ── Parsing helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _extract(text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_answer(text: str) -> str:
        m = re.search(r"\b([A-D])\b", text)
        return m.group(1) if m else text.strip()

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
