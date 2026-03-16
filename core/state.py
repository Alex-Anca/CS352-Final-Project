from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SharedState:
    left_gesture: str = "none"   # "fist" | "open_palm" | "pinch" | "none"
    right_gesture: str = "none"
    current_beat: int = 0
    position_in_beat: int = 0
    eighth_note_pos: int = 0
    running: bool = True
    frame: Optional[Any] = None   # latest camera frame, written by gesture thread
    result: Optional[Any] = None  # latest MediaPipe result, written by gesture thread
    prev_beat: Optional[int] = None   # beat just before current, for line drawing
    tightness: float = 0.0            # 0.0=sequential, 1.0=always cluster-jump; written by gesture thread
    lpf_cutoff: float = 1.0           # 1.0=open, 0.0=fully filtered; written by gesture thread
    cluster_labels: Optional[Any] = None  # np.ndarray (N,) int - written once at startup
    pca_coords: Optional[Any] = None      # np.ndarray (N, 2) float - written once at startup
    active_clusters: set = field(default_factory=lambda: {0})  # 0-indexed; written by gesture thread
    cluster_pending: int = 0       # finger count being held for selection (0 = none)
    cluster_pending_t: float = 0.0 # elapsed hold time in seconds
