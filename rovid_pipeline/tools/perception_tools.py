"""
Robust-TO — Perception Tools  (Section 3.2 / Table 17)

The paper's full perception library (Table 17) lists FIVE perception tools:

    detect_objects   (cost 0.50) — object detection with bounding boxes
    caption_frame    (cost 0.30) — dense captioning of frame content
    track_temporal   (cost 0.70) — multi-frame object/action tracking
    recognize_action (cost 0.60) — action recognition with temporal context
    read_text        (cost 0.25) — OCR for in-video text

WHAT THIS RELEASE SHIPS
-----------------------
Per the paper's plug-and-play design (Section 3.2: "the routing logic depends
only on semantic-type and disturbance-profile interfaces: new tools can be
registered by declaring these two properties, making the framework readily
extensible"), this open-source release ships the THREE components needed to run
the pipeline end-to-end:

    * detect_objects   — backed by the APE detection service (ape_tools/)
    * read_text        — OCR via the host VLM
    * the APE service  — the detection backend itself (ape_tools/)

The remaining perception tools (caption_frame, track_temporal,
recognize_action) are provided as PLUG-AND-PLAY templates only.  They raise
NotImplementedError until you supply a backend.  Because every tool conforms to
the same unified (result, confidence) contract (Eq. 4) and declares a semantic
type + disturbance profile for routing (Table 18), adding a tool is purely
additive and requires no change to the orchestration layer — register it in
`build_perception_tools()` (or pass it in via `extra_tools=`) and the router
(stages/perception.py) will resolve it by name at runtime.

NOTE on the detection backbone (paper Section B.1 discrepancy)
--------------------------------------------------------------
Paper Section B.1 states detect_objects "wraps a GroundingDINO-T model".  This
release instead wraps APE (Aligning and Prompting Everything), which is the
detector actually used in the released code.  The intrinsic-confidence
definition still follows the paper (B.1): the mean detection confidence over the
returned bounding boxes, with boxes below the 0.3 score threshold discarded.
If you reproduce with GroundingDINO-T, swap the backend in DetectObjects._run;
the (result, confidence) contract is unchanged.
"""

from __future__ import annotations
import pickle
import socket
import struct
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import ToolBase, ToolResult


# ── Type alias ────────────────────────────────────────────────────────────────
InferenceFn = Callable[[str, Optional[np.ndarray]], str]
"""fn(prompt: str, frames: Optional[np.ndarray]) -> str"""


# Score threshold for discarding low-confidence detections (paper Section B.1 /
# Table 16: GroundingDINO-T threshold 0.3).
DETECTION_SCORE_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Socket helpers (length-prefixed framing) — used by detect_objects / APE
# ─────────────────────────────────────────────────────────────────────────────

def send_msg(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed message so large payloads are never truncated."""
    sock.sendall(struct.pack(">Q", len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Optional[bytes]:
    """Receive a length-prefixed message; returns None if the peer closed."""
    header = _recv_exact(sock, 8)
    if header is None:
        return None
    (length,) = struct.unpack(">Q", header)
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ─────────────────────────────────────────────────────────────────────────────
# detect_objects  (RELEASED — APE backend)
# ─────────────────────────────────────────────────────────────────────────────

class DetectObjects(ToolBase):
    """
    Object detection with bounding boxes  (Table 17, cost 0.50).

    Delegates to the APE service (ape_tools/ape_service.py) over a TCP socket
    with length-prefixed framing.

    Intrinsic confidence (paper Section B.1): the mean detection score over the
    returned bounding boxes, with boxes scoring below DETECTION_SCORE_THRESHOLD
    (0.3) discarded.  When the APE service is unreachable, the call fails
    gracefully (result=None, c_intrinsic handled as a failed call by base.py).

    Result format: list of per-frame detection strings, e.g.
        ["cat: [10, 20, 50, 60]; dog: [80, 30, 40, 40]", ...]
    """

    name = "detect_objects"

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        save_dir: Optional[str] = None,
        score_threshold: float = DETECTION_SCORE_THRESHOLD,
    ):
        self.host = host
        self.port = port
        self.score_threshold = score_threshold
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="robustto_detect_")

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Optional[List[str]], float]:
        import os
        from PIL import Image

        if len(frames) == 0:
            return None, 0.0

        os.makedirs(self.save_dir, exist_ok=True)
        frame_paths: List[str] = []
        for i, f in enumerate(frames):
            p = os.path.join(self.save_dir, f"det_frame_{i}.png")
            Image.fromarray(f).save(p)
            frame_paths.append(p)

        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(60.0)
            client.connect((self.host, self.port))
            send_msg(client, pickle.dumps((frame_paths, sub_query)))
            raw = recv_msg(client)
            client.close()
            if raw is None:
                return None, 0.0
            # APE returns: list[ list[ (label, [x,y,w,h], score) ] ] per frame
            per_frame = pickle.loads(raw)
        except Exception:
            return None, 0.0

        det_strings: List[str] = []
        kept_scores: List[float] = []
        for dets in per_frame:
            parts = []
            for label, box, score in dets:
                if score < self.score_threshold:
                    continue
                kept_scores.append(float(score))
                box_str = ", ".join(str(int(v)) for v in box)
                parts.append(f"{label}: [{box_str}]")
            det_strings.append("; ".join(parts))

        # c_intrinsic = mean detection score over kept boxes (paper Section B.1)
        c_intrinsic = float(np.mean(kept_scores)) if kept_scores else 0.0
        return det_strings, c_intrinsic


# ─────────────────────────────────────────────────────────────────────────────
# read_text  (RELEASED — VLM/OCR backend)
# ─────────────────────────────────────────────────────────────────────────────

class ReadText(ToolBase):
    """
    OCR for in-video text  (Table 17, cost 0.25).

    Paper Section B.1 wraps PaddleOCR with c_intrinsic = mean character-level
    recognition confidence.  This release reads text through the host VLM in an
    OCR/text-reading prompting mode (so the release stays runnable without a
    separate OCR dependency); to reproduce the paper exactly, swap PaddleOCR in
    here and report its character-level confidence as c_intrinsic.

    Result format: list of per-frame transcribed-text strings.
    """

    name = "read_text"

    def __init__(self, inference_fn: InferenceFn):
        self._infer = inference_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Optional[List[str]], float]:
        if len(frames) == 0:
            return None, 0.0
        texts: List[str] = []
        for frame in frames:
            prompt = (
                "Read any visible text in this video frame (signs, labels, "
                f"license plates, captions). Focus on: {sub_query}\n"
                "Return ONLY the transcribed text, or 'NONE' if no text is present."
            )
            texts.append(self._infer(prompt, frame[np.newaxis]))
        # c_intrinsic: fraction of frames yielding non-trivial text (stand-in for
        # PaddleOCR's mean character-level confidence; see docstring).
        non_empty = sum(1 for t in texts if t and t.strip().upper() not in ("", "NONE"))
        c_intrinsic = non_empty / max(len(frames), 1)
        return texts, float(c_intrinsic)


# ─────────────────────────────────────────────────────────────────────────────
# Plug-and-play templates  (NOT shipped — implement your own backend)
# ─────────────────────────────────────────────────────────────────────────────

class PluggableTool(ToolBase):
    """
    Base for plug-and-play perception tools that are part of the paper's library
    (Table 17) but are NOT shipped in this open-source release.

    To add one, subclass and implement _run() returning (result, c_intrinsic),
    then register the instance via build_perception_tools(extra_tools={name: obj})
    or add it directly to the returned dict.  The unified confidence coupling
    (Eq. 4), the semantic-type/disturbance routing (Table 18), and the
    confidence-cost reward (Eqs. 5-10) all work unchanged once the tool is
    registered — nothing in the orchestration layer needs to be touched.

    Example
    -------
        class MyCaptioner(PluggableTool):
            name = "caption_frame"
            def _run(self, frames, sub_query, disturbance_scores):
                cap = my_captioning_model(frames, sub_query)   # your backend
                return cap, my_confidence_score                # in [0, 1]

        tools = build_perception_tools(
            vlm_fn, extra_tools={"caption_frame": MyCaptioner()}
        )
    """

    name = ""

    def _run(self, frames, sub_query, disturbance_scores):
        raise NotImplementedError(
            f"Perception tool '{self.name}' is a plug-and-play template and is "
            f"not shipped in this release. Provide a backend by subclassing "
            f"PluggableTool (see its docstring) and registering it via "
            f"build_perception_tools(extra_tools={{'{self.name}': <your_tool>}})."
        )


class CaptionFrame(PluggableTool):
    """caption_frame (Table 17, cost 0.30). Plug-and-play — supply a backend."""
    name = "caption_frame"


class TrackTemporal(PluggableTool):
    """track_temporal (Table 17, cost 0.70). Plug-and-play — supply a backend."""
    name = "track_temporal"


class RecognizeAction(PluggableTool):
    """recognize_action (Table 17, cost 0.60). Plug-and-play — supply a backend."""
    name = "recognize_action"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_perception_tools(
    inference_fn: InferenceFn,
    ape_host: str = "0.0.0.0",
    ape_port: int = 9999,
    extra_tools: Optional[Dict[str, ToolBase]] = None,
) -> Dict[str, ToolBase]:
    """
    Construct the perception tool library.

    Ships the RELEASED tools (detect_objects via APE, read_text via the VLM).
    Pass `extra_tools={name: ToolBase}` to register additional perception tools
    (e.g. your own caption_frame / track_temporal / recognize_action backends);
    they are merged in and resolved by the router by name — no other change
    needed.  This is the paper's plug-and-play extension point (Section 3.2).

    The returned dict is keyed by `tool.name`.
    """
    tools: Dict[str, ToolBase] = {
        "detect_objects": DetectObjects(host=ape_host, port=ape_port),
        "read_text":      ReadText(inference_fn),
    }
    if extra_tools:
        for name, tool in extra_tools.items():
            tools[name] = tool
    return tools
