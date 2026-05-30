"""
generate_mlvu.py — MLVU evaluation with RoVid pipeline

Replaces the original Video-RAG inference core with the three-stage
RoVid pipeline (Scout → Perceive → Contemplate).

Usage:
    python evals/generate_mlvu.py \
        --data_dir MLVU/MLVU_Dev.json \
        --video_folder MLVU/video \
        --output_dir results \
        --max_frames 64
"""

import argparse
import json
import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# ── RoVid pipeline ────────────────────────────────────────────────────────────
from rovid_pipeline import RoVidPipeline, process_video
from rovid_pipeline.rovid_pipeline import agent_fn, vlm_inference_fn


# ─────────────────────────────────────────────────────────────────────────────
# MLVU Dataset  (kept from original — only loading logic, no inference)
# ─────────────────────────────────────────────────────────────────────────────

class MLVU(Dataset):
    """MLVU multiple-choice video QA dataset loader."""

    def __init__(self, data_dir: str, video_folder: str):
        self.video_folder = video_folder
        self.data_list = []

        with open(data_dir, "r") as f:
            json_data = json.load(f)

        for data in json_data:
            self.data_list.append({
                "task_type":   data["question_type"],
                "data":        data,
                "question_id": data["video"] + "_" + data["question"],
                "candidates":  data["candidates"],
                "answer":      data["answer"],
            })

    def __len__(self):
        return len(self.data_list)

    def __str__(self):
        from collections import Counter
        counts = Counter(d["task_type"] for d in self.data_list)
        lines  = [f"  {v:>4}  {k}" for k, v in sorted(counts.items())]
        return f"MLVU ({len(self.data_list)} items):\n" + "\n".join(lines)

    def qa_template(self, data: dict) -> str:
        """Format question + options as a single query string."""
        q = f"Question: {data['question']}\nOptions:\n"
        for idx, c in enumerate(data["candidates"]):
            q += f"({chr(ord('A') + idx)}) {c}\n"
        return q.rstrip()

    def __getitem__(self, idx: int) -> dict:
        entry      = self.data_list[idx]
        video_path = os.path.join(self.video_folder, entry["data"]["video"])
        question   = self.qa_template(entry["data"])
        gt_letter  = chr(ord("A") + entry["candidates"].index(entry["answer"]))
        return {
            "video":       video_path,
            "question":    question,
            "task_type":   entry["task_type"],
            "question_id": entry["question_id"],
            "answer":      gt_letter,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="MLVU/MLVU_Dev.json")
    p.add_argument("--video_folder",  default="MLVU/video")
    p.add_argument("--output_dir",    default="results")
    p.add_argument("--max_frames",    type=int, default=64)
    p.add_argument("--k_simple",      type=int, default=4)
    p.add_argument("--k_complex",     type=int, default=16)
    return p.parse_args()


def _save(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(results: list) -> dict:
    """Compute per-task and overall accuracy from saved results."""
    from collections import defaultdict
    correct_by_task = defaultdict(int)
    total_by_task   = defaultdict(int)

    for r in results:
        t = r.get("task_type", "unknown")
        total_by_task[t]   += 1
        if r.get("response", "").strip().upper() == r.get("answer", "").strip().upper():
            correct_by_task[t] += 1

    acc = {}
    for t in total_by_task:
        acc[t] = correct_by_task[t] / total_by_task[t]

    total   = sum(total_by_task.values())
    correct = sum(correct_by_task.values())
    acc["overall"] = correct / total if total > 0 else 0.0
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_out = os.path.join(args.output_dir, "generate_mlvu.json")

    dataset = MLVU(args.data_dir, args.video_folder)
    print(dataset)

    # Resume from checkpoint
    results      = []
    process_set  = set()
    if os.path.exists(json_out):
        with open(json_out, "r", encoding="utf-8") as f:
            results = json.load(f)
        process_set = {r["question_id"] for r in results}
    print(f"[MLVU] Resuming — {len(results)}/{len(dataset)} done")

    # ── Init RoVid once ───────────────────────────────────────────────────────
    pipeline = RoVidPipeline(
        agent_fn  = agent_fn,
        vlm_fn    = vlm_inference_fn,
        k_simple  = args.k_simple,
        k_complex = args.k_complex,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    for example in tqdm(dataset, desc="MLVU"):
        if example["question_id"] in process_set:
            continue

        try:
            frames, _, _ = process_video(
                example["video"], max_frames=args.max_frames, force_sample=True
            )
        except Exception as e:
            print(f"  [WARN] Video load failed: {e}")
            results.append({
                "question_id": example["question_id"],
                "task_type":   example["task_type"],
                "response":    "",
                "answer":      example["answer"],
            })
            _save(results, json_out)
            continue

        try:
            out = pipeline.run(
                frames       = frames,
                query        = example["question"],
                ground_truth = example["answer"],
            )
            response = out["answer"]
            reward   = out["reward"].R_total
        except Exception as e:
            print(f"  [WARN] Pipeline error: {e}")
            response, reward = "", 0.0

        results.append({
            "question_id": example["question_id"],
            "task_type":   example["task_type"],
            "response":    response,
            "answer":      example["answer"],
            "reward":      reward,
        })
        _save(results, json_out)

    # ── Print accuracy ────────────────────────────────────────────────────────
    acc = compute_accuracy(results)
    print("\n── MLVU Accuracy ──")
    for task, a in sorted(acc.items()):
        print(f"  {task:<30} {a*100:.1f}%")
    print(f"  {'OVERALL':<30} {acc['overall']*100:.1f}%")
    print(f"\nSaved to {json_out}")


if __name__ == "__main__":
    main()
