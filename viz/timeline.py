"""
Circular beat timeline renderer.

Beats are arranged in chronological order around a circle (beat 0 at the top,
going clockwise).  Each render() call draws:

  - Notches colored by similarity to the current beat (bright = similar)
  - Fading cubic bezier arcs for recent non-sequential jumps
  - A green bezier to the most-similar beat (grows with tightness)
  - The current beat as a large bright white notch
"""

import math
from collections import deque

import cv2
import numpy as np


_NOTCH_LEN    = 26   # normal notch inward length (px)
_NOTCH_OUTER  = 5    # normal notch extra length outside the ring (px)
_BEZIER_STEPS = 40
_BEZIER_PULL  = 0.38  # control points pulled this fraction toward center

_COLOR_PURPLE = (160, 40,  120)  # BGR: solid purple (dissimilar beats)
_COLOR_PINK   = (210, 80,  255)  # BGR: bright pink (top-5 similar beats)
_TOP_N_SIMILAR = 5


class TimelineRenderer:
    def __init__(self, num_beats, D, beat_chunk_lengths, canvas_size=(480, 480)):
        self._N = num_beats
        self._D = D
        self._chunk_lens = beat_chunk_lengths
        H, W = canvas_size
        self._H, self._W = H, W
        self._cx, self._cy = W // 2, H // 2
        self._radius = min(H, W) // 2 - 36

        # Precompute angle and ring pixel position for each beat
        self._angles = [2 * math.pi * i / num_beats - math.pi / 2
                        for i in range(num_beats)]
        self._beat_px = [
            (int(self._cx + self._radius * math.cos(a)),
             int(self._cy + self._radius * math.sin(a)))
            for a in self._angles
        ]

        # Static background: just the ring (notches drawn per-frame with colors)
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.circle(bg, (self._cx, self._cy), self._radius, (35, 35, 35), 1, cv2.LINE_AA)
        self._static_bg = bg

        # Jump arc history: each entry is [from_beat, to_beat, age_in_frames]
        self._jump_history: deque = deque(maxlen=8)
        self._last_beat: int | None = None

    # ------------------------------------------------------------------
    # Helpers

    def _notch_pts(self, beat_idx, length_in, outer_extra=2):
        """Return (outer, inner) pixel coords for a radial notch."""
        a = self._angles[beat_idx]
        cos_a, sin_a = math.cos(a), math.sin(a)
        outer = (int(self._cx + (self._radius + outer_extra) * cos_a),
                 int(self._cy + (self._radius + outer_extra) * sin_a))
        inner = (int(self._cx + (self._radius - length_in) * cos_a),
                 int(self._cy + (self._radius - length_in) * sin_a))
        return outer, inner

    def _bezier_pts(self, from_b, to_b):
        """Cubic bezier from from_b to to_b, control points pulled toward center."""
        p0 = np.array(self._beat_px[from_b], dtype=float)
        p3 = np.array(self._beat_px[to_b],   dtype=float)
        c  = np.array([self._cx, self._cy],   dtype=float)
        p1 = p0 + _BEZIER_PULL * (c - p0)
        p2 = p3 + _BEZIER_PULL * (c - p3)
        pts = []
        for i in range(_BEZIER_STEPS + 1):
            t = i / _BEZIER_STEPS
            pt = ((1-t)**3 * p0
                  + 3*(1-t)**2 * t * p1
                  + 3*(1-t) * t**2 * p2
                  + t**3 * p3)
            pts.append([[int(pt[0]), int(pt[1])]])
        return np.array(pts, dtype=np.int32)

    # ------------------------------------------------------------------

    def render(self, state) -> np.ndarray:
        canvas = self._static_bg.copy()
        current_beat = state.current_beat

        # --- Similarity colors: top-5 similar = pink, rest = purple ---
        distances = self._D[current_beat].copy()
        distances[current_beat] = np.inf
        top5 = set(np.argsort(distances)[:_TOP_N_SIMILAR].tolist())

        for i in range(self._N):
            if i == current_beat:
                continue
            color = _COLOR_PINK if i in top5 else _COLOR_PURPLE
            outer, inner = self._notch_pts(i, _NOTCH_LEN, _NOTCH_OUTER)
            cv2.line(canvas, outer, inner, color, 1, cv2.LINE_AA)

        # --- Jump arc history ---
        if (self._last_beat is not None
                and current_beat != self._last_beat
                and current_beat != (self._last_beat + 1) % self._N):
            self._jump_history.append([self._last_beat, current_beat, 0])
        self._last_beat = current_beat

        for entry in self._jump_history:
            entry[2] += 1

        fade_frames = 40
        for from_b, to_b, age in self._jump_history:
            if age >= fade_frames:
                continue
            a = 1.0 - age / fade_frames
            color = (int(60 * a), int(190 * a), int(255 * a))
            if 0 <= from_b < self._N and 0 <= to_b < self._N:
                cv2.polylines(canvas, [self._bezier_pts(from_b, to_b)],
                              False, color, 1, cv2.LINE_AA)

        # --- Green bezier to most-similar beat (scales with tightness) ---
        tightness = state.tightness
        if tightness > 0.03 and 0 <= current_beat < self._N:
            row2 = self._D[current_beat].copy()
            row2[current_beat] = np.inf
            target = int(np.argmin(row2))
            a = min(tightness * 1.4, 1.0)
            t_color = (int(60 * a), int(210 * a), int(110 * a))
            cv2.polylines(canvas, [self._bezier_pts(current_beat, target)],
                          False, t_color, 1, cv2.LINE_AA)

        # --- Current beat: large white notch drawn last (always on top) ---
        if 0 <= current_beat < self._N:
            outer, inner = self._notch_pts(current_beat, 45, 13)
            cv2.line(canvas, outer, inner, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas
