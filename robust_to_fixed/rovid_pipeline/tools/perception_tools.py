"""
RoVid — Perception Tools (Section 3.1 / Table 17)

Implements the FIVE perception tools from the paper's tool library (Table 17),
all sharing the unified (result, confidence) interface from `base.py`:

    detect_objects   (cost 0.50) — object detection with bounding boxes (APE)
    caption_frame    (cost 0.30) — dense captioning of frame content (VLM)
    track_temporal   (cost 0.70) — multi-frame object/action tracking
    recognize_action (cost 0.60) — action recognition with temporal context
    read_text        (cost 0.25) — OCR for in-video text

FIX (Inconsistency A — incomplete tool library):
    The previous version implemented only detect_objects and caption_frame.
    The paper (Table 17, Table 18, and case studies in Tabs. 7/14/15) describes
    and uses all FIVE perception tools, and the disturbance-aware routing table
    (Table 18) is defined over all of them.  All five are now implemented.

    track_temporal, recognize_action, and read_text wrap the host VLM in
    task-specific prompting modes (in lieu of the paper's ByteTrack /
    VideoMAE-v2 / PaddleOCR backbones, which require heavy external deps).
    Each conforms to the unified (result, confidence) contract so that the
    confidence-coupling (Eq. 4) and routing (Table 18) work uniformly across
    the whole library, exactly as the paper specifies, while staying runnable.
"""

from __future__ import annotations
import pickle
import socket
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import ToolBase, ToolResult


# ── Type alias ────────────────────────────────────────────────────────────────
InferenceFn = Callable[[str, Optional[np.ndarray]], str]
"""fn(prompt: str, frames: Optional[np.ndarray]) -> str"""


# ─────────────────────────────────────────────────────────────────────────────
# detect_objects
# ─────────────────────────────────────────────────────────────────────────────

class DetectObjects(ToolBase):
    """
    Object detection with bounding boxes.

    Delegates to the APE service (`ape_tools/ape_service.py`) via TCP socket,
    matching the existing socket protocol used elsewhere in the pipeline.

    Returns
    -------
    result : list of per-frame detection strings, e.g.
             ["cat: [10, 20, 50, 60]; dog: [80, 30, 40, 40]", ...]
    """

    name = "detect_objects"

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        save_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="rovid_detect_")

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[List[str], float]:
        import os
        from PIL import Image

        os.makedirs(self.save_dir, exist_ok=True)
        frame_paths: List[str] = []
        for i, f in enumerate(frames):
            p = os.path.join(self.save_dir, f"det_frame_{i}.png")
            Image.fromarray(f).save(p)
            frame_paths.append(p)

        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(30.0)
            client.connect((self.host, self.port))
            client.send(pickle.dumps((frame_paths, sub_query)))
            chunks = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            client.close()
            raw = b"".join(chunks)
            det_results: List[str] = pickle.loads(raw)
        except Exception:
            det_results = [""] * len(frames)

        # c_intrinsic: fraction of frames with non-empty detections
        non_empty = sum(1 for r in det_results if r and len(r) > 0)
        c_intrinsic = non_empty / max(len(frames), 1)

        return det_results, c_intrinsic


# ─────────────────────────────────────────────────────────────────────────────
# caption_frame
# ─────────────────────────────────────────────────────────────────────────────

class CaptionFrame(ToolBase):
    """
    Dense captioning of frame content via the VLM.

    Tolerates spatial corruption better than fine-grained detectors, making it
    the preferred fallback under blur-type degradation
    (Section 1.4, disturbance-conditioned tool selection).

    Returns
    -------
    result : list of per-frame caption strings
    """

    name = "caption_frame"

    def __init__(self, inference_fn: InferenceFn):
        self._infer = inference_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[List[str], float]:
        captions: List[str] = []
        for frame in frames:
            prompt = (
                f"Describe this video frame in detail, focusing on: {sub_query}\n"
                "Provide a dense, specific description of objects, actions, and scene."
            )
            caption = self._infer(prompt, frame[np.newaxis])  # single frame as video
            captions.append(caption)

        # c_intrinsic: average normalised length (longer captions → richer description)
        avg_len = np.mean([min(len(c) / 200.0, 1.0) for c in captions])
        return captions, float(avg_len)


# ─────────────────────────────────────────────────────────────────────────────
# track_temporal
# ─────────────────────────────────────────────────────────────────────────────

class TrackTemporal(ToolBase):
    """
    Multi-frame object/action tracking (Table 17, cost 0.70).

    Paper wraps a ByteTrack tracker; c_intrinsic is the mean IoU of matched
    tracklets across frames (Section B.1).  Here we delegate to the host VLM
    in a temporal-reasoning prompting mode and derive c_intrinsic from output
    completeness, conforming to the unified (result, confidence) interface.
    """

    name = "track_temporal"

    def __init__(self, inference_fn: InferenceFn):
        self._infer = inference_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[str, float]:
        if len(frames) == 0:
            return None, 0.0
        prompt = (
            "Track the movement/trajectory of the relevant objects across these "
            f"consecutive video frames. Focus on: {sub_query}\n"
            "Report the temporal order and motion of each tracked target."
        )
        result = self._infer(prompt, frames)
        # c_intrinsic: completeness proxy (longer, non-empty tracking output → higher)
        c_intrinsic = float(min(len(result) / 150.0, 1.0)) if result else 0.0
        return result, c_intrinsic


# ─────────────────────────────────────────────────────────────────────────────
# recognize_action
# ─────────────────────────────────────────────────────────────────────────────

class RecognizeAction(ToolBase):
    """
    Action recognition with temporal context (Table 17, cost 0.60).

    Paper wraps a VideoMAE-v2 classifier; c_intrinsic is the softmax probability
    of the top-1 predicted action class (Section B.1).  Here we delegate to the
    host VLM in an action-recognition prompting mode.
    """

    name = "recognize_action"

    def __init__(self, inference_fn: InferenceFn):
        self._infer = inference_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[str, float]:
        if len(frames) == 0:
            return None, 0.0
        prompt = (
            "Recognize the primary action or event occurring in this video clip. "
            f"Focus on: {sub_query}\n"
            "Return the single most likely action label and a brief justification."
        )
        result = self._infer(prompt, frames)
        c_intrinsic = float(min(len(result) / 120.0, 1.0)) if result else 0.0
        return result, c_intrinsic


# ─────────────────────────────────────────────────────────────────────────────
# read_text
# ─────────────────────────────────────────────────────────────────────────────

class ReadText(ToolBase):
    """
    OCR for in-video text (Table 17, cost 0.25).

    Paper wraps PaddleOCR; c_intrinsic is the mean character-level recognition
    confidence (Section B.1).  Here we delegate to the host VLM in an
    OCR/text-reading prompting mode.
    """

    name = "read_text"

    def __init__(self, inference_fn: InferenceFn):
        self._infer = inference_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[List[str], float]:
        if len(frames) == 0:
            return None, 0.0
        texts: List[str] = []
        for frame in frames:
            prompt = (
                "Read any visible text in this video frame (signs, labels, license "
                f"plates, captions). Focus on: {sub_query}\n"
                "Return ONLY the transcribed text, or 'NONE' if no text is present."
            )
            texts.append(self._infer(prompt, frame[np.newaxis]))
        # c_intrinsic: fraction of frames yielding non-trivial text
        non_empty = sum(1 for t in texts if t and t.strip().upper() not in ("", "NONE"))
        c_intrinsic = non_empty / max(len(frames), 1)
        return texts, c_intrinsic


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_perception_tools(
    inference_fn: InferenceFn,
    ape_host: str = "0.0.0.0",
    ape_port: int = 9999,
) -> Dict[str, ToolBase]:
    """
    Construct the full perception tool library (all 5 tools from Table 17).

    The returned dict is keyed by `tool.name` so that the dispatcher in
    `stages/perceive.py` can resolve tool selections by name.
    """
    return {
        "detect_objects":   DetectObjects(host=ape_host, port=ape_port),
        "caption_frame":    CaptionFrame(inference_fn),
        "track_temporal":   TrackTemporal(inference_fn),
        "recognize_action": RecognizeAction(inference_fn),
        "read_text":        ReadText(inference_fn),
    }
