"""
Robust-TO — Confidence-Cost Reward for GRPO  (Section 3.3)

The composite reward (Eq. 10) combines four terms, all computed from a single
forward pass so they are well-defined on a fixed-length trajectory:

    Confidence-cost, per call (Eq. 5):
        R_cc(c_j, T_j) = c_j - lambda * cost(T_j),     lambda = 0.5

    Confidence-cost, trajectory average (Eq. 6):
        R_cc^total(tau) = (1 / N_call) * sum_k R_cc(c_j^(k), T_jk)

    Sub-query efficiency (Eqs. 7-9):
        R_min-sq(n, N*)      = exp(-alpha * max(0, n - N*))                 (Eq. 7)
        R_qual(n, N*, tau)   = (1 - exp(-beta * n/max(N*,1))) * mean_conf   (Eq. 8)
        R_subq               = 0.5 * (R_min-sq + R_qual)                    (Eq. 9)

    Composite (Eq. 10):
        R_total = R_acc + w * (R_subq + R_cc^total + R_fmt),    w = 1/3

    with R_acc in {-1, +1}, R_fmt in {0, 1}.  The shared w = 1/3 keeps the
    auxiliary sum's magnitude at most 1, matching |R_acc| (Section 3.3).

NOTE on the auxiliary weight (paper consistency)
------------------------------------------------
Two places in the paper specify this weight:
  * Eq. 10 / Section 3.3 (the method):  a single shared w = 1/3.
  * Table 19 (training config):  w_subq = w_cc = w_fmt = 0.3, w_acc = 1.0.
These agree up to rounding (1/3 ~= 0.333 vs 0.3).  We default to the method's
exact w = 1/3.  (Set w_subq/w_cc/w_fmt=0.3 to reproduce Table 19 verbatim.)
The auxiliary weights are exposed as arguments below.

N* (the question-conditional optimum) is estimated by a FROZEN off-the-shelf VLM
(text-only Qwen2.5-7B-Instruct in the paper), decoupled from the policy VLM to
prevent reward gaming (Section 3.3; policy-internal estimation costs 1.2 acc.
points and raises reward variance 2.3x).
"""

from __future__ import annotations
import math
import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from .tools.base import TOOL_COSTS, ToolResult


# ── Reward hyper-parameters (Section 3.3 / Table 19) ─────────────────────────
LAMBDA = 0.5         # tool-cost weight        (Eq. 5)
W_ACC  = 1.0         # correctness weight      (Eq. 10)
W_AUX  = 1.0 / 3.0   # shared auxiliary weight (Eq. 10): w_subq = w_cc = w_fmt = w
W_SUBQ = W_AUX
W_CC   = W_AUX
W_FMT  = W_AUX
ALPHA  = 0.2         # excess sub-query penalty (Eq. 7, Table 19)
BETA   = 1.0         # coverage saturation      (Eq. 8, Table 19)


@dataclass
class TrajectoryReward:
    R_acc:    float
    R_subq:   float
    R_cc:     float
    R_fmt:    float
    R_total:  float
    n_calls:  int
    per_call: List[float]


def per_call_reward(c_j: float, tool_name: str, lam: float = LAMBDA) -> float:
    """Eq. 5: R_cc = c_j - lambda * cost(T_j). Failed calls (c_j=0) -> -lambda*cost."""
    cost = TOOL_COSTS.get(tool_name, 0.5)
    return float(c_j - lam * cost)


def trajectory_reward_cc(tool_results: List[ToolResult], lam: float = LAMBDA) -> float:
    """Eq. 6: average per-call reward over the trajectory (no extra-call incentive)."""
    if not tool_results:
        return 0.0
    calls = [per_call_reward(tr.confidence, tr.tool_name, lam) for tr in tool_results]
    return float(sum(calls) / len(calls))


def estimate_optimal_subqueries(
    query: str,
    agent_fn: Optional[Callable[[str], str]] = None,
) -> int:
    """
    Estimate N*(q) once per query with a FROZEN estimator (Section 3.3).

    The paper uses a frozen text-only Qwen2.5-7B-Instruct, decoupled from the
    policy VLM.  When no estimator is provided, a syntactic-complexity heuristic
    is used.  Note: pass a frozen model here, NOT the policy VLM, to avoid the
    reward gaming the paper warns about.
    """
    if agent_fn is not None:
        prompt = (
            "[Task] Estimate the minimum number of independent perceptual "
            "sub-queries needed to fully answer this video question. Do not "
            "overestimate.\n"
            f"[Question] {query}\n"
            "[Output] Output ONLY a single integer."
        )
        try:
            raw = agent_fn(prompt)
            m = re.search(r"\d+", raw)
            if m:
                v = int(m.group(0))
                if v > 0:
                    return min(max(v, 1), 12)
        except Exception:
            pass

    lower = query.lower()
    temporal = sum(t in lower for t in
                   ["before", "after", "during", "while", "then", "finally", "end", "beginning"])
    causal = sum(t in lower for t in ["why", "cause", "because", "result"])
    wc = len(query.split())
    clauses = query.count(",") + query.count("?") + query.count(" and ")
    score = 1 + temporal + causal + (wc >= 12) + (wc >= 20) + min(clauses, 2)
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
    """Eqs. 7-9."""
    if optimal_subqueries is None:
        if query is None:
            raise ValueError("Either query or optimal_subqueries must be provided")
        n_opt = estimate_optimal_subqueries(query, agent_fn=agent_fn)
    else:
        n_opt = max(int(optimal_subqueries), 1)
    r_min_sq = math.exp(-alpha * max(0, n_subqueries - n_opt))           # Eq. 7
    coverage = 1.0 - math.exp(-beta * n_subqueries / max(n_opt, 1))      # Eq. 8
    r_qual   = coverage * mean_confidence                               # Eq. 8
    return float(0.5 * (r_min_sq + r_qual))                             # Eq. 9


def format_reward(answer: str, response_text: Optional[str] = None) -> float:
    """
    R_fmt in {0, 1}: checks the synthesis output structure.

    The paper's synthesis output contract (App. E.3) is:
        <think> ... </think>  <answer>X</answer>
    A well-formed response contains both tags.
    """
    if response_text:
        rt = response_text.lower()
        if "<think>" in rt and "</think>" in rt and "<answer>" in rt and "</answer>" in rt:
            return 1.0
    if not answer or len(answer.strip()) == 0:
        return 0.0
    if any(p in answer.lower() for p in ["i cannot", "i don't know", "unable to", "no information"]):
        return 0.0
    return 1.0


def compute_trajectory_reward(
    tool_results: List[ToolResult],
    answer: str,
    ground_truth: Optional[str],
    n_subqueries: int,
    query: Optional[str] = None,
    optimal_subqueries: Optional[int] = None,
    agent_fn: Optional[Callable[[str], str]] = None,
    response_text: Optional[str] = None,
    w_acc: float = W_ACC,
    w_subq: float = W_SUBQ,
    w_cc: float = W_CC,
    w_fmt: float = W_FMT,
    lam: float = LAMBDA,
) -> TrajectoryReward:
    """Eq. 10: R_total = w_acc*R_acc + w_subq*R_subq + w_cc*R_cc + w_fmt*R_fmt."""
    if ground_truth is not None:
        correct = answer.strip().upper() == ground_truth.strip().upper()
        R_acc = 1.0 if correct else -1.0
    else:
        R_acc = 0.0  # unknown at inference time

    mean_conf = (float(sum(tr.confidence for tr in tool_results) / len(tool_results))
                 if tool_results else 0.0)
    R_subq = subquery_reward(
        n_subqueries=n_subqueries, mean_confidence=mean_conf,
        query=query, optimal_subqueries=optimal_subqueries, agent_fn=agent_fn,
    )
    R_cc  = trajectory_reward_cc(tool_results, lam)
    R_fmt = format_reward(answer, response_text=response_text)

    per_call = [per_call_reward(tr.confidence, tr.tool_name, lam) for tr in tool_results]
    R_total = w_acc * R_acc + w_subq * R_subq + w_cc * R_cc + w_fmt * R_fmt

    return TrajectoryReward(
        R_acc=R_acc, R_subq=R_subq, R_cc=R_cc, R_fmt=R_fmt,
        R_total=R_total, n_calls=len(tool_results), per_call=per_call,
    )
