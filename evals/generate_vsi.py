"""
generate_vsi.py — VSI-Bench evaluation with the Robust-TO pipeline.

VSI-Bench (paper Section B.3) is indoor spatial intelligence from ego-centric
navigation, with four multiple-choice tasks:
    RDist = Relative Distance, RDir = Relative Direction,
    RP    = Route Planning,    AO   = Appearance Order.
Metric: accuracy (%).

The expected metadata JSON is a list of records, each like:
    {
      "video": "relative/path/clip.mp4",
      "question": "....",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "C",
      "task": "RDist"          # one of RDist / RDir / RP / AO
    }
Adjust load_records() to match your local VSI-Bench release if its schema
differs; everything else is benchmark-agnostic.

Usage:
    python evals/generate_vsi.py \
        --data_path /path/to/VSI-Bench \
        --json_meta evals/vsi_meta.json \
        --output_dir results --max_frames 32
"""
import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

from rovid_pipeline import RobustTOPipeline, process_video
from rovid_pipeline.rovid_pipeline import agent_fn, vlm_inference_fn

TASKS = ["RDist", "RDir", "RP", "AO"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="/path/to/VSI-Bench")
    p.add_argument("--json_meta",  default="evals/vsi_meta.json")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--k_simple",   type=int, default=4)
    p.add_argument("--k_complex",  type=int, default=12)
    return p.parse_args()


def load_records(json_meta):
    with open(json_meta, "r", encoding="utf-8") as f:
        return json.load(f)


def format_question(rec) -> str:
    return rec["question"] + "\n" + " ".join(rec.get("options", []))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.json_meta)

    pipeline = RobustTOPipeline(
        agent_fn=agent_fn, vlm_fn=vlm_inference_fn,
        k_simple=args.k_simple, k_complex=args.k_complex,
    )

    per_task = defaultdict(lambda: [0, 0])
    outputs = []
    for rec in tqdm(records, desc="VSI-Bench"):
        video_path = os.path.join(args.data_path, rec["video"])
        try:
            frames, _, _ = process_video(video_path, max_frames=args.max_frames, force_sample=True)
        except Exception as e:
            print(f"skip {video_path}: {e}")
            continue

        gt = rec.get("answer")
        result = pipeline.run(frames, format_question(rec), ground_truth=gt)
        pred = result["answer"]
        task = rec.get("task", "RDist")
        correct = (gt is not None and pred.strip().upper() == gt.strip().upper())
        per_task[task][0] += int(correct)
        per_task[task][1] += 1
        outputs.append({
            "video": rec["video"], "task": task,
            "prediction": pred, "answer": gt, "correct": correct,
        })

    summary = {}
    total_c = total_n = 0
    for t in TASKS:
        c, n = per_task[t]
        summary[t] = round(100.0 * c / n, 2) if n else None
        total_c += c
        total_n += n
    summary["Avg"] = round(100.0 * total_c / total_n, 2) if total_n else None

    with open(os.path.join(args.output_dir, "vsi_predictions.json"), "w") as f:
        json.dump(outputs, f, indent=2)
    with open(os.path.join(args.output_dir, "vsi_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Accuracy (%):", summary)


if __name__ == "__main__":
    main()
