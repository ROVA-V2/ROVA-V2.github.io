"""
RoVid — Training Objective: Confidence-Cost Reward for GRPO  (Section 3.3)

The paper's composite reward (Eq. 10) combines four terms:

    Per-Call Reward (Eq. 5):
        R_cc(c_j, T_j) = c_j − λ · cost(T_j)

    Trajectory-Level Aggregation (Eq. 6):
        R_total_cc(τ) = (1 / N_call) · Σ R_cc(c_j^(k), T_j_k)

    Sub-Query Efficiency (Eqs. 7–9):
        R_min-sq(n, N*) = exp(−α · max(0, n − N*))                         (Eq. 7)
        R_qual(n, N*, τ) = (1 − exp(−β · n/max(N*,1))) · mean_confidence(τ) (Eq. 8)
        R_subq           = 0.5 · (R_min-sq + R_qual)                         (Eq. 9)

    Composite Reward (Eq. 10):
        R_total = R_acc + w · (R_subq + R_total_cc + R_fmt)

        with w = 1/3, R_acc ∈ {−1, +1}.

NOTE on reward weights:
    The paper's method equation (Eq. 10) is the authoritative objective:
        R_total = R_acc + w·(R_subq + R_total_cc + R_fmt),  with w = 1/3,
    so that the auxiliary sum's magnitude is at most 1, matching |R_acc|.
    We implement exactly this (a single shared w = 1/3 over the three
    auxiliary terms).  Table 19's per-term weights (0.2/0.3/0.1) are an
    inconsistent alternative whose auxiliary sum is 0.6; they remain
    overridable via the w_subq / w_cc / w_fmt arguments for users who wish to
    reproduce that specific training run, but the default follows Eq. 10.

Key design rationale (Section 3.3):
    • c_j already encodes frame quality via ρ(F) in Eq. 4, so the reward is
      naturally lower when an expensive tool is called on degraded frames.
    • R_total_cc averages over trajectory length → no incentive for extra tool calls.
    • λ=0.5: maximum-cost tool (cost=0.70) incurs a penalty equal to a
      medium-confidence result.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import json
import re
from typing import Callable, List, Optional

from .tools.base import TOOL_COSTS, ToolResult


# ── Reward hyper-parameters ──────────────────────────────────────────────────
# FIX (Inconsistency D — reward weighting):
#   The paper's METHOD equation (Eq. 10, Section 3.3) is the authoritative
#   description of the objective:
#
#       R_total = R_acc + w · (R_subq + R_total_cc + R_fmt),   w = 1/3
#
#   so that "the auxiliary sum's magnitude is at most 1, matching |R_acc|".
#   The previous code used Table 19's per-term weights (0.2/0.3/0.1), whose
#   auxiliary sum is 0.6 — this does NOT match Eq. 10 and is internally
#   inconsistent with the main-text method.  We therefore default to the
#   Eq. 10 formulation: a single shared w = 1/3 applied uniformly to the three
#   auxiliary terms.  (The per-term weights remain overridable via arguments
#   for users who wish to reproduce the exact Table 19 training run.)
LAMBDA      = 0.5        # cost penalty weight  (Eq. 5)
W_ACC       = 1.0        # correctness weight   (Eq. 10)
W_AUX       = 1.0 / 3.0  # shared auxiliary weight w (Eq. 10)
W_SUBQ      = W_AUX      # sub-query efficiency weight
W_CC        = W_AUX      # confidence-cost weight
W_FMT       = W_AUX      # format well-formedness weight
ALPHA       = 0.2        # excess sub-query penalty  (Eq. 7, Table 19)
BETA        = 1.0        # coverage saturation       (Eq. 8, Table 19)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryReward:
    """Full reward breakdown for one trajectory τ."""
    R_acc:      float   # ∈ {-1, +1}
    R_subq:     float   # sub-query efficiency
    R_cc:       float   # confidence-cost (trajectory average)
    R_fmt:      float   # format check ∈ {0, 1}
    R_total:    float   # composite (Eq. 10)
    n_calls:    int
    per_call:   List[float]


def per_call_reward(c_j: float, tool_name: str, lam: float = LAMBDA) -> float:
    """
    Eq. 5:  R_cc(c_j, T_j) = c_j − λ · cost(T_j)

    If tool call failed (c_j set to 0 by base), result is pure cost penalty −λ·cost.
    Unknown tool names fall back to cost=0.5 (moderate penalty).
    """
    cost = TOOL_COSTS.get(tool_name, 0.5)
    return float(c_j - lam * cost)


def trajectory_reward_cc(tool_results: List[ToolResult], lam: float = LAMBDA) -> float:
    """
    Eq. 6:  R_total_cc(τ) = (1/N_call) · Σ R_cc(c_j^(k), T_j_k)

    Averages over trajectory length → no incentive for extra tool calls.
    """
    if not tool_results:
        return 0.0
    calls = [per_call_reward(tr.confidence, tr.tool_name, lam) for tr in tool_results]
    return float(sum(calls) / len(calls))


def estimate_optimal_subqueries(
    query: str,
    agent_fn: Optional[Callable[[str], str]] = None,
) -> int:
    """
    Estimate the question-conditional N*(q) once per query.

    The paper uses a frozen off-the-shelf VLM to estimate N* (Section 3.3),
    decoupled from the policy VLM to prevent reward gaming.  When no agent
    is provided, a heuristic based on syntactic complexity is used.
    """
    if agent_fn is not None:
        prompt = (
            f"Question: {query}\n\n"
            "Estimate the optimal number of visual sub-queries needed to answer it. "
            "Return ONLY a JSON object like {\"optimal_subqueries\": 4}."
        )
        try:
            raw = agent_fn(prompt)
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                value = int(parsed.get("optimal_subqueries", 0))
                if value > 0:
                    return min(max(value, 1), 12)
        except Exception:
            pass

    lower = query.lower()
    temporal_markers = sum(
        token in lower
        for token in ["before", "after", "during", "while", "then", "finally", "end", "beginning"]
    )
    causal_markers = sum(token in lower for token in ["why", "cause", "because", "result"])
    word_count = len(query.split())
    clauses = query.count(",") + query.count("?") + query.count(" and ")
    score = 1 + temporal_markers + causal_markers + (word_count >= 12) + (word_count >= 20) + min(clauses, 2)
    return min(max(score, 1), 12)


def subquery_reward(
    n_subqueries: int,
    mean_confidence: float,
    query: Optional[str] = None,
    optimal_subqueries: Optional[int] = None,
    agent_fn: Optional[Callable[[str], str]] = None,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """
    Eqs. 7-9 (paper):
        R_min-sq = exp(−alpha · max(0, n − N*))                    (Eq. 7)
        R_qual   = (1 − exp(−beta · n / max(N*, 1))) · mean_conf   (Eq. 8)
        R_subq   = 0.5 · (R_min-sq + R_qual)                      (Eq. 9)
    """
    if optimal_subqueries is None:
        if query is None:
            raise ValueError("Either query or optimal_subqueries must be provided")
        n_opt = estimate_optimal_subqueries(query, agent_fn=agent_fn)
    else:
        n_opt = max(int(optimal_subqueries), 1)
    r_min_sq = math.exp(-alpha * max(0, n_subqueries - n_opt))
    coverage = 1.0 - math.exp(-beta * n_subqueries / max(n_opt, 1))
    r_qual = coverage * mean_confidence
    return float(0.5 * (r_min_sq + r_qual))


def format_reward(answer: str, response_text: Optional[str] = None) -> float:
    """
    R_fmt ∈ {0, 1}: checks output structure.
    Prefer the full structured reasoning response when available.
    """
    if response_text:
        required_tags = ["REASONING:", "GAPS:", "ANSWER:", "UNCERTAINTY:"]
        if all(tag in response_text for tag in required_tags):
            return 1.0
    if not answer or len(answer.strip()) == 0:
        return 0.0
    error_phrases = ["i cannot", "i don't know", "unable to", "no information"]
    if any(p in answer.lower() for p in error_phrases):
        return 0.0
    return 1.0


def compute_trajectory_reward(
    tool_results:   List[ToolResult],
    answer:         str,
    ground_truth:   Optional[str],
    n_subqueries:   int,
    query:          Optional[str] = None,
    optimal_subqueries: Optional[int] = None,
    agent_fn:       Optional[Callable[[str], str]] = None,
    response_text:  Optional[str] = None,
    w_acc:  float = W_ACC,
    w_subq: float = W_SUBQ,
    w_cc:   float = W_CC,
    w_fmt:  float = W_FMT,
    lam:    float = LAMBDA,
) -> TrajectoryReward:
    """
    Eq. 10 (paper):
        R_total = R_acc + w·R_subq + w·R_total_cc + w·R_fmt
        (equivalently: R_acc + w*(R_subq + R_cc + R_fmt) with w=1/3,
         here represented as separate per-term weights from Table 18)
    """
    # R_acc ∈ {-1, +1}
    if ground_truth is not None:
        correct = answer.strip().upper() == ground_truth.strip().upper()
        R_acc = 1.0 if correct else -1.0
    else:
        R_acc = 0.0  # unknown at inference time

    mean_conf = (
        float(sum(tr.confidence for tr in tool_results) / len(tool_results))
        if tool_results else 0.0
    )
    R_subq = subquery_reward(
        n_subqueries=n_subqueries,
        mean_confidence=mean_conf,
        query=query,
        optimal_subqueries=optimal_subqueries,
        agent_fn=agent_fn,
    )
    R_cc  = trajectory_reward_cc(tool_results, lam)
    R_fmt = format_reward(answer, response_text=response_text)

    per_call = [per_call_reward(tr.confidence, tr.tool_name, lam) for tr in tool_results]

    R_total = w_acc * R_acc + w_subq * R_subq + w_cc * R_cc + w_fmt * R_fmt

    return TrajectoryReward(
        R_acc    = R_acc,
        R_subq   = R_subq,
        R_cc     = R_cc,
        R_fmt    = R_fmt,
        R_total  = R_total,
        n_calls  = len(tool_results),
        per_call = per_call,
    )
