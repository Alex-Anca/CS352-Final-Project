"""
Comprehensive gesture detection tests using the Jester dataset.

Samples 25 clips per Jester label, runs MediaPipe HandLandmarker (IMAGE mode)
on 5 middle frames per clip, and caches all detected landmarks in setUpClass.
Every test then asserts a detection rate (or mean finger count) over that pool
of ~100–120 real-world frames per label — giving statistical coverage rather
than cherry-picked single-clip checks.

Thresholds are calibrated against actual detection rates on this dataset.
Jester frames are small (100×176 px) and not always ideal-angle shots, so
positive thresholds are conservative.  Negative thresholds allow for label
ambiguity (e.g. "Swiping Down" sometimes starts/ends with fingers slightly
curled, so 50 % open-palm is the realistic floor).

Label notes:
  "Zooming Out/In With Full Hand" — these are spreading-finger motions that
  look like open palms at mid-clip; they are good open-palm positives but
  not reliable pinch positives across all clips.  One low-threshold pinch
  test is kept to verify that is_pinch() CAN fire in natural footage.

  "Thumb Up" — four fingers fully curled, only thumb up; this is the
  cleanest fist-positive and open-palm-negative available in Jester.
"""

import csv
import os
import random
import sys
import unittest

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core.handsoff import count_extended_fingers, is_fist, is_open_palm, is_pinch

DATASET_DIR = os.path.join(ROOT, "jester_dataset", "Train")
CSV_PATH    = os.path.join(ROOT, "jester_dataset", "Train.csv")
MODEL_PATH  = os.path.join(ROOT, "models", "hand_landmarker.task")

N_CLIPS  = 25   # clips sampled per label
N_FRAMES = 5    # middle frames used per clip

NEEDED_LABELS = [
    "Stop Sign",
    "Shaking Hand",
    "Swiping Right",
    "Swiping Up",
    "Swiping Down",
    "Pushing Hand Away",
    "Thumb Down",
    "Thumb Up",
    "Zooming Out With Full Hand",
    "Zooming In With Full Hand",
    "Pulling Two Fingers In",
    "Pushing Two Fingers Away",
]


def _load_label_index():
    """Return {label: [video_id, ...]} from Train.csv."""
    index = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            index.setdefault(row["label"], []).append(int(row["video_id"]))
    return index


def _middle_frame_indices(video_id, n):
    """Return n frame indices centered on the clip midpoint."""
    folder = os.path.join(DATASET_DIR, str(video_id))
    frames = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    total  = len(frames)
    if total == 0:
        return []
    mid   = total // 2
    start = max(0, mid - n // 2)
    end   = min(total, start + n)
    return [int(frames[i].replace(".jpg", "")) for i in range(start, end)]


class TestGestures(unittest.TestCase):
    """Integration tests: Jester frames → MediaPipe landmarks → gesture functions."""

    @classmethod
    def setUpClass(cls):
        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
        )
        cls.landmarker = vision.HandLandmarker.create_from_options(opts)

        label_index = _load_label_index()
        rng = random.Random(42)

        # cls.lms[label] = flat list of all detected landmark lists for that label
        cls.lms: dict[str, list] = {}
        for label in NEEDED_LABELS:
            all_ids = label_index.get(label, [])
            sample  = rng.sample(all_ids, min(N_CLIPS, len(all_ids)))
            lms = []
            for vid in sample:
                for fi in _middle_frame_indices(vid, N_FRAMES):
                    path = os.path.join(DATASET_DIR, str(vid), f"{fi:05d}.jpg")
                    if not os.path.exists(path):
                        continue
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = cls.landmarker.detect(mp_img)
                    if result.hand_landmarks:
                        lms.append(result.hand_landmarks[0])
            cls.lms[label] = lms

    @classmethod
    def tearDownClass(cls):
        cls.landmarker.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rate(self, label, fn):
        """Detection rate (0–100 int) of fn across all cached frames for label."""
        lms = self.lms[label]
        if not lms:
            return 0
        return sum(fn(lm) for lm in lms) * 100 // len(lms)

    def _avg_count(self, label):
        """Mean count_extended_fingers across all cached frames for label."""
        lms = self.lms[label]
        if not lms:
            return 0.0
        return sum(count_extended_fingers(lm) for lm in lms) / len(lms)

    def _require_detections(self, label, minimum=10):
        self.assertGreaterEqual(
            len(self.lms[label]), minimum,
            f"Too few hand detections for '{label}': {len(self.lms[label])}"
        )

    def _pos(self, label, fn, threshold):
        self._require_detections(label)
        rate = self._rate(label, fn)
        self.assertGreaterEqual(
            rate, threshold,
            f"[FALSE NEGATIVE] '{label}': {fn.__name__} fired {rate}% < required {threshold}%"
        )

    def _neg(self, label, fn, threshold):
        self._require_detections(label)
        rate = self._rate(label, fn)
        self.assertLessEqual(
            rate, threshold,
            f"[FALSE POSITIVE] '{label}': {fn.__name__} fired {rate}% > allowed {threshold}%"
        )

    # ------------------------------------------------------------------
    # is_open_palm — positives
    # ------------------------------------------------------------------

    def test_open_palm_pos_stop_sign(self):
        """Stop Sign: hand held open and upright."""
        self._pos("Stop Sign", is_open_palm, 55)

    def test_open_palm_pos_shaking_hand(self):
        """Shaking Hand: sustained open palm throughout clip."""
        self._pos("Shaking Hand", is_open_palm, 55)

    def test_open_palm_pos_swiping_right(self):
        """Swiping Right: flat open hand moving laterally."""
        self._pos("Swiping Right", is_open_palm, 70)

    def test_open_palm_pos_swiping_up(self):
        """Swiping Up: open palm moving upward."""
        self._pos("Swiping Up", is_open_palm, 70)

    def test_open_palm_pos_swiping_down(self):
        """Swiping Down: open palm moving downward."""
        self._pos("Swiping Down", is_open_palm, 50)

    def test_open_palm_pos_pushing_hand_away(self):
        """Pushing Hand Away: open palm pushed forward."""
        self._pos("Pushing Hand Away", is_open_palm, 55)

    def test_open_palm_pos_zoom_in_full(self):
        """Zooming In With Full Hand: fingers spreading outward, mostly open."""
        self._pos("Zooming In With Full Hand", is_open_palm, 65)

    # ------------------------------------------------------------------
    # is_open_palm — negatives (false-positive checks)
    # ------------------------------------------------------------------

    def test_open_palm_neg_thumb_down(self):
        """Thumb Down is a closed fist — should not trigger open palm."""
        self._neg("Thumb Down", is_open_palm, 20)

    def test_open_palm_neg_thumb_up(self):
        """Thumb Up: only thumb extended, four fingers curled — not open palm."""
        self._neg("Thumb Up", is_open_palm, 5)

    # ------------------------------------------------------------------
    # is_fist — positives
    # ------------------------------------------------------------------

    def test_fist_pos_thumb_down(self):
        """Thumb Down: four fingers fully curled → strong fist signal."""
        self._pos("Thumb Down", is_fist, 70)

    def test_fist_pos_thumb_up(self):
        """Thumb Up: same fist shape, only direction of thumb differs."""
        self._pos("Thumb Up", is_fist, 60)

    # ------------------------------------------------------------------
    # is_fist — negatives
    # ------------------------------------------------------------------

    def test_fist_neg_stop_sign(self):
        """Stop Sign (open hand) should not trigger fist."""
        self._neg("Stop Sign", is_fist, 15)

    def test_fist_neg_swiping_right(self):
        """Swiping Right (open hand moving laterally) should not trigger fist."""
        self._neg("Swiping Right", is_fist, 10)

    # ------------------------------------------------------------------
    # is_pinch — positive (low threshold: Jester lacks a clean pinch class)
    # ------------------------------------------------------------------

    def test_pinch_pos_zoom_out_full(self):
        """
        Zooming Out With Full Hand: starts from a full-hand pinch and spreads
        outward. Mid-clip frames often still satisfy the pinch condition.
        Threshold is low (10 %) because the spread happens quickly and only
        the pinch-start frames qualify — but the test confirms is_pinch()
        fires in real footage, not just synthetic data.
        """
        self._pos("Zooming Out With Full Hand", is_pinch, 10)

    # ------------------------------------------------------------------
    # is_pinch — negatives
    # ------------------------------------------------------------------

    def test_pinch_neg_swiping_right(self):
        """Flat open hand should not trigger pinch."""
        self._neg("Swiping Right", is_pinch, 15)

    def test_pinch_neg_thumb_down(self):
        """Fist (no fingers extended) does not satisfy pinch's other-finger check."""
        self._neg("Thumb Down", is_pinch, 10)

    # ------------------------------------------------------------------
    # count_extended_fingers
    # ------------------------------------------------------------------

    def test_count_two_pulling(self):
        """Pulling Two Fingers In: index+middle extended → avg ≈ 2."""
        self._require_detections("Pulling Two Fingers In")
        avg = self._avg_count("Pulling Two Fingers In")
        self.assertAlmostEqual(avg, 2.0, delta=0.7,
            msg=f"Pulling Two Fingers In: expected ~2 fingers, got {avg:.2f}")

    def test_count_two_pushing(self):
        """Pushing Two Fingers Away: index+middle extended → avg ≈ 2."""
        self._require_detections("Pushing Two Fingers Away")
        avg = self._avg_count("Pushing Two Fingers Away")
        self.assertAlmostEqual(avg, 2.0, delta=0.7,
            msg=f"Pushing Two Fingers Away: expected ~2 fingers, got {avg:.2f}")

    def test_count_four_open_palm(self):
        """Swiping Right (open palm): all four non-thumb fingers extended → avg ≥ 3."""
        self._require_detections("Swiping Right")
        avg = self._avg_count("Swiping Right")
        self.assertGreaterEqual(avg, 3.0,
            msg=f"Swiping Right (open palm): expected ≥3 fingers, got {avg:.2f}")

    def test_count_zero_fist(self):
        """Thumb Down (fist): mostly no non-thumb fingers extended → avg ≤ 0.7."""
        self._require_detections("Thumb Down")
        avg = self._avg_count("Thumb Down")
        self.assertLessEqual(avg, 0.7,
            msg=f"Thumb Down (fist): expected ≤0.7 fingers, got {avg:.2f}")

    def test_count_zero_thumb_up(self):
        """Thumb Up: four fingers curled → avg ≤ 0.7 extended (thumb excluded)."""
        self._require_detections("Thumb Up")
        avg = self._avg_count("Thumb Up")
        self.assertLessEqual(avg, 0.7,
            msg=f"Thumb Up: expected ≤0.7 fingers, got {avg:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
