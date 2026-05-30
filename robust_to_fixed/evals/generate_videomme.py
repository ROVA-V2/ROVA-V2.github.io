"""
generate_videomme.py — VideoMME evaluation with RoVid pipeline

Replaces the original Video-RAG inference core with the three-stage
RoVid pipeline (Scout → Perceive → Contemplate).

Usage:
    python evals/generate_videomme.py \
        --data_path /path/to/Video-MME/data \
        --output_dir results \
        --max_frames 64
"""

import argparse
import json
import os

from tqdm import tqdm

# ── RoVid pipeline ────────────────────────────────────────────────────────────
from rovid_pipeline import RoVidPipeline, process_video
from rovid_pipeline.rovid_pipeline import agent_fn, vlm_inference_fn


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="/path/to/Video-MME/data")
    p.add_argument("--json_meta",  default="evals/videomme_json_file.json")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--k_simple",   type=int, default=4)
    p.add_argument("--k_complex",  type=int, default=16)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_question(q: dict) -> str:
    """Combine question + options into a single natural-language query."""
    return q["question"] + "\n" + " ".join(q["options"])


def _save(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_out = os.path.join(args.output_dir, "generate_videomme.json")

    with open(args.json_meta, "r", encoding="utf-8") as f:
        mme_data = json.load(f)

    # Resume from checkpoint
    rep_list = []
    if os.path.exists(json_out):
        with open(json_out, "r", encoding="utf-8") as f:
            rep_list = json.load(f)
    start = len(rep_list)
    print(f"[VideoMME] Resuming from {start}/{len(mme_data)}")

    # ── Init RoVid once ───────────────────────────────────────────────────────
    pipeline = RoVidPipeline(
        agent_fn  = agent_fn,
        vlm_fn    = vlm_inference_fn,
        k_simple  = args.k_simple,
        k_complex = args.k_complex,
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    for item in tqdm(mme_data[start:], desc="VideoMME"):
        video_path = os.path.join(args.data_path, item["url"] + ".mp4")
        content    = item.copy()

        try:
            frames, _, _ = process_video(video_path, max_frames=args.max_frames,
                                         force_sample=True)
        except Exception as e:
            print(f"  [WARN] Video load failed ({video_path}): {e}")
            rep_list.append(content)
            _save(rep_list, json_out)
            continue

        for q in content["questions"]:
            query = format_question(q)
            gt    = q.get("answer", "")
            try:
                out = pipeline.run(frames=frames, query=query, ground_truth=gt)
                q["response"]     = out["answer"]
                q["reasoning"]    = out["reasoning"]
                q["reward"]       = out["reward"].R_total
                q["n_tool_calls"] = out["n_tool_calls"]
                q["stage1_info"]  = out["stage1_info"]
            except Exception as e:
                print(f"  [WARN] Pipeline error: {e}")
                q["response"] = ""

        rep_list.append(content)
        _save(rep_list, json_out)

    print(f"Done. Saved to {json_out}")


if __name__ == "__main__":
    main()
