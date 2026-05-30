"""Robust-TO (project RoVA-V2) — confidence-aware tool orchestration for robust video understanding."""
from .reward import compute_trajectory_reward, TrajectoryReward

try:
    from .rovid_pipeline import RobustTOPipeline, RoVidPipeline, process_video
except Exception:
    RobustTOPipeline = None
    RoVidPipeline = None
    process_video = None

__all__ = [
    "RobustTOPipeline",
    "RoVidPipeline",          # backward-compatible alias
    "process_video",
    "compute_trajectory_reward",
    "TrajectoryReward",
]
