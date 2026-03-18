from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SharedState:
    left_gesture: str = "none"   # "fist" | "open_palm" | "pinch" | "none"
    right_gesture: str = "none"
    current_beat: int = 0
    position_in_beat: int = 0
    running: bool = True
    frame: Optional[Any] = None   # latest camera frame, written by gesture thread
    result: Optional[Any] = None  # latest MediaPipe result, written by gesture thread
    prev_beat: Optional[int] = None   # beat just before current, for line drawing
    lpf_cutoff: float = 1.0           # 1.0=open, 0.0=fully filtered; auto-driven by reverb_depth
    tightness: float = 0.0            # 0.0=sequential, 1.0=always jump; both-pinch elastic control
    reverb_depth: float = 0.0         # 0.0–1.0; left hand size proxy; EMA-smoothed
    reverb_tail: float = 0.5          # 0.0=short decay, 1.0=long decay; left hand finger spread
    delay_depth: float = 0.0          # 0.0–1.0; right hand size proxy; controls delay feedback
    delay_subdivision: int = 0        # 0=off 1=quarter 2=eighth 3=dotted-8th 4=sixteenth
    cluster_labels: Optional[Any] = None  # np.ndarray (N,) int - written once at startup
    pca_coords: Optional[Any] = None      # np.ndarray (N, 2) float - written once at startup
    active_clusters: set = field(default_factory=lambda: {0})  # 0-indexed; set to all at startup
