"""
generate_longvideobench.py — LongVideoBench evaluation with RoVid pipeline

Replaces the original Video-RAG inference core with the three-stage
RoVid pipeline (Scout → Perceive → Contemplate).

Requires the `longvideobench` package:
    pip install longvideobench

Usage:
    python evals/generate_longvideobench.py \
        --data_path /path/to/LongVideoBenchData \
        --split val \
        --output_dir results \
        --max_frames 64
"""

import argparse
import json
import os

from tqdm import tqdm

# Dataset loader (from the longvideobench package — unchanged)
from longvideobench import LongVideoBenchDataset

# ── RoVid pipeline ────────────────────────────────────────────────────────────
from rovid_pipeline import RoVidPipeline, process_video
from rovid_pipeline.rovid_pipeline import agent_fn, vlm_inference_fn


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

LETTERS = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="/path/to/LongVideoBenchData")
    p.add_argument("--split",      default="val",
                   choices=["val", "test"],
                   help="'val' uses lvb_val.json; 'test' uses lvb_test_wo_gt.json")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--k_simple",   type=int, default=4)
    p.add_argument("--k_complex",  type=int, default=16)
    return p.parse_args()


def _save(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Format LVB item into a natural-language query
# ─────────────────────────────────────────────────────────────────────────────

def format_question(item: dict) -> str:
    """Build a query string from question + candidates."""
    q = f"Question: {item['question']}\n"
    for i, c in enumerate(item["candidates"]):
        q += f"{LETTERS[i]}. {c} "
    q += "\nSelect the best answer (A/B/C/D/E)."
    return q.strip()


def item_gt(item: dict) -> str:
    """Ground-truth answer letter, or '' for test set."""
    if "correct_choice" in item:
        return LETTERS[item["correct_choice"]]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helper (val set only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(results: list) -> dict:
    correct = sum(
        1 for r in results
        if r.get("gt") and r.get("response", "").strip().upper() == r["gt"].upper()
    )
    total = sum(1 for r in results if r.get("gt"))
    return {"overall": correct / total if total else 0.0, "correct": correct, "total": total}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_meta = "lvb_val.json" if args.split == "val" else "lvb_test_wo_gt.json"
    json_out  = os.path.join(args.output_dir, f"generate_longvideobench_{args.split}.json")

    # Load LVB dataset  (LongVideoBenchDataset returns .data — a list of dicts)
    lvb = LongVideoBenchDataset(
        args.data_path,
        json_meta,
        max_num_frames=args.max_frames,
    )
    mme_data = lvb.data
    print(f"[LVB] {len(mme_data)} items, split={args.split}")

    # Resume from checkpoint
    rep_list   = []
    done_ids   = set()
    if os.path.exists(json_out):
        with open(json_out, "r", encoding="utf-8") as f:
            rep_list = json.load(f)
        done_ids = {r["video"] for r in rep_list}
    start = len(rep_list)
    print(f"[LVB] Resuming from {start}/{len(mme_data)}")

    # ── Init RoVid once ───────────────────────────────────────────────────────
    pipeline = RoVidPipeline(
        agent_fn  = agent_fn,
        vlm_fn    = vlm_inference_fn,
        k_simple  = args.k_simple,
        k_complex = args.k_complex,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    for item in tqdm(mme_data, desc=f"LVB-{args.split}"):
        vid_id = item.get("video_id", item.get("video", ""))
        if vid_id in done_ids:
            continue

        video_path = os.path.join(args.data_path, "videos", item["video_path"])
        query      = format_question(item)
        gt         = item_gt(item)

        try:
            frames, _, _ = process_video(
                video_path, max_frames=args.max_frames, force_sample=True
            )
        except Exception as e:
            print(f"  [WARN] Video load failed ({video_path}): {e}")
            rep_list.append({
                "video":    vid_id,
                "response": "",
                "gt":       gt,
            })
            _save(rep_list, json_out)
            continue

        try:
            out = pipeline.run(
                frames       = frames,
                query        = query,
                ground_truth = gt or None,
            )
            response = out["answer"]
            reward   = out["reward"].R_total
        except Exception as e:
            print(f"  [WARN] Pipeline error: {e}")
            response, reward = "", 0.0

        rep_list.append({
            "video":       vid_id,
            "response":    response,
            "gt":          gt,
            "reward":      reward,
            "n_tool_calls": out.get("n_tool_calls", 0) if "out" in dir() else 0,
            "stage1_info": out.get("stage1_info", {}) if "out" in dir() else {},
        })
        _save(rep_list, json_out)

    # ── Print accuracy (val set) ──────────────────────────────────────────────
    if args.split == "val":
        acc = compute_accuracy(rep_list)
        print(f"\n── LVB-val Accuracy ──")
        print(f"  {acc['correct']}/{acc['total']}  =  {acc['overall']*100:.1f}%")

    print(f"\nSaved to {json_out}")


if __name__ == "__main__":
    main()
