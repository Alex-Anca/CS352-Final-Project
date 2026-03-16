import math
import time
from collections import deque
from statistics import mode

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.handsoff import is_open_palm, is_fist, is_pinch, count_extended_fingers
from core.state import SharedState


def classify_gesture(lm):
    """Return gesture string for a landmark list."""
    if is_pinch(lm):
        return "pinch"
    elif is_fist(lm):
        return "fist"
    elif is_open_palm(lm):
        return "open_palm"
    n = count_extended_fingers(lm)
    if 1 <= n <= 4:
        return f"fingers_{n}"
    return "none"


def run_gesture_thread(state: SharedState) -> None:
    """Camera loop: detect gestures, smooth via rolling buffer, write to state.
    Does NOT do any display - frame and result are written to state for the
    main thread to render."""
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    start = time.time()

    left_buf: deque = deque(maxlen=4)
    right_buf: deque = deque(maxlen=4)

    prev_left = "none"
    prev_right = "none"

    # pinch-knob tracking - incremental encoder style (frame-to-frame deltas)
    left_prev_angle = None
    right_prev_angle = None

    # cluster selection: 2s hold of fingers_N on right hand
    finger_hold_count = 0
    finger_hold_start = 0.0
    finger_committed = False

    # per-hand landmark cache for angle computation
    left_lm = None
    right_lm = None

    try:
        while state.running:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            detected_left = "none"
            detected_right = "none"
            left_lm = None
            right_lm = None

            if result and result.hand_landmarks:
                for lm, handedness in zip(result.hand_landmarks, result.handedness):
                    mp_hand = handedness[0].category_name
                    user_hand = "left" if mp_hand == "Left" else "right"
                    gesture = classify_gesture(lm)

                    if user_hand == "left":
                        detected_left = gesture
                        left_lm = lm
                    else:
                        detected_right = gesture
                        right_lm = lm

            left_buf.append(detected_left)
            right_buf.append(detected_right)

            smoothed_left = mode(left_buf)
            smoothed_right = mode(right_buf)

            # --- LEFT PINCH KNOB: controls lpf_cutoff ---
            if smoothed_left == "pinch" and left_lm is not None:
                current_angle = math.atan2(
                    left_lm[9].y - left_lm[0].y,
                    left_lm[9].x - left_lm[0].x,
                )
                if left_prev_angle is not None:
                    delta = current_angle - left_prev_angle
                    delta = (delta + math.pi) % (2 * math.pi) - math.pi
                    state.lpf_cutoff = max(0.0, min(1.0, state.lpf_cutoff + delta / (math.pi / 3)))
                left_prev_angle = current_angle
            else:
                left_prev_angle = None

            # --- RIGHT PINCH KNOB: controls tightness ---
            if smoothed_right == "pinch" and right_lm is not None:
                current_angle = math.atan2(
                    right_lm[9].y - right_lm[0].y,
                    right_lm[9].x - right_lm[0].x,
                )
                if right_prev_angle is not None:
                    delta = current_angle - right_prev_angle
                    delta = (delta + math.pi) % (2 * math.pi) - math.pi
                    state.tightness = max(0.0, min(1.0, state.tightness - delta / (math.pi / 3)))
                right_prev_angle = current_angle
            else:
                right_prev_angle = None

            # --- RIGHT HAND FINGER COUNT: cluster selection (2s hold) ---
            right_count = int(smoothed_right[-1]) if smoothed_right.startswith("fingers_") else 0
            if right_count != finger_hold_count:
                finger_hold_count = right_count
                finger_hold_start = time.time()
                finger_committed = False
            if right_count >= 1 and not finger_committed:
                elapsed = time.time() - finger_hold_start
                state.cluster_pending = right_count
                state.cluster_pending_t = elapsed
                if elapsed >= 2.0:
                    state.active_clusters = {right_count - 1}
                    finger_committed = True
                    print(f"Cluster selected: {right_count}")
            else:
                state.cluster_pending = 0
                state.cluster_pending_t = 0.0

            state.left_gesture = smoothed_left
            state.right_gesture = smoothed_right

            prev_left = smoothed_left
            prev_right = smoothed_right

            # write frame + result for main thread to display
            state.frame = frame
            state.result = result

    finally:
        landmarker.close()
        cap.release()
