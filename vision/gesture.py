import time
from collections import deque
from statistics import mode

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.handsoff import is_open_palm, is_fist, is_pinch, count_extended_fingers
from core.state import SharedState


# ── Gesture Config ────────────────────────────────────────────────────────────
DEPTH_MIN        = 0.14   # wrist→midMCP 2D span at comfortable distance (depth 0.0)
DEPTH_MAX        = 0.26   # wrist→midMCP 2D span when hand is pushed forward (depth 1.0)
DEPTH_EMA_ALPHA  = 0.20   # smoothing factor (lower = slower/smoother)
SUBDIV_BUF_LEN   = 8      # debounce window size
SUBDIV_MIN_VOTES = 5      # frames of agreement required to commit
PINCH_DIST_MIN   = 0.05   # index-tip distance when hands nearly touching → tightness 1.0
PINCH_DIST_MAX   = 0.55   # index-tip distance when arms spread wide   → tightness 0.0
TIGHTNESS_ALPHA  = 0.08   # EMA smoothing for tightness
TAIL_EMA_ALPHA   = 0.12   # EMA smoothing for reverb tail (finger count changes discretely)
AUTO_TIGHTNESS   = 0.0    # initial tightness (fully tight / sequential)


def _smoothstep(x):
    """Clamp x to [0,1] and apply smoothstep — eases smoothly into both extremes."""
    t = max(0.0, min(1.0, x))
    return t * t * (3.0 - 2.0 * t)


def classify_gesture(lm):
    """Return gesture string for a landmark list."""
    if is_fist(lm):
        return "fist"
    elif is_pinch(lm):
        return "pinch"
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

    # per-hand landmark cache
    left_lm = None
    right_lm = None

    # LEFT HAND: depth EMA + tail EMA
    depth_ema: float = 0.0
    tail_ema: float = 0.5

    # RIGHT HAND: depth EMA state + finger-count debounce buffer
    right_depth_ema: float = 0.0
    subdiv_buf: deque = deque(maxlen=SUBDIV_BUF_LEN)

    # BOTH HANDS PINCH: elastic tightness EMA
    tightness_ema: float = AUTO_TIGHTNESS

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

            # --- LEFT HAND: 2D hand size → reverb depth (skip when pinching) ---
            if left_lm is not None and smoothed_left != "pinch":
                raw_size = ((left_lm[9].x - left_lm[0].x) ** 2 +
                            (left_lm[9].y - left_lm[0].y) ** 2) ** 0.5
                normalized = max(0.0, min(1.0,
                    (raw_size - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)))
                depth_ema = depth_ema * (1 - DEPTH_EMA_ALPHA) + normalized * DEPTH_EMA_ALPHA
            # lost or pinching: hold last value
            state.reverb_depth = _smoothstep(depth_ema)

            # --- LEFT HAND: finger count → reverb tail length ---
            if left_lm is not None and smoothed_left != "pinch":
                raw_tail = count_extended_fingers(left_lm) / 4.0  # 0.0 (fist) – 1.0 (open)
                tail_ema = tail_ema * (1 - TAIL_EMA_ALPHA) + raw_tail * TAIL_EMA_ALPHA
            # lost or pinching: hold last value
            state.reverb_tail = _smoothstep(tail_ema)

            # --- RIGHT HAND: 2D hand size → delay feedback depth (skip when pinching) ---
            if right_lm is not None and smoothed_right != "pinch":
                raw_size = ((right_lm[9].x - right_lm[0].x) ** 2 +
                            (right_lm[9].y - right_lm[0].y) ** 2) ** 0.5
                normalized = max(0.0, min(1.0,
                    (raw_size - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)))
                right_depth_ema = right_depth_ema * (1 - DEPTH_EMA_ALPHA) + normalized * DEPTH_EMA_ALPHA
            # lost or pinching: hold last value
            state.delay_depth = _smoothstep(right_depth_ema)

            # --- RIGHT HAND: finger count → delay subdivision (skip when pinching) ---
            # fingers_N and open_palm both use the actual finger count (thumb excluded);
            # open_palm fires when thumb is also up alongside 4 fingers, so treat it the
            # same way. Fist and none map to 0 (delay off).
            if right_lm is not None and smoothed_right != "pinch":
                if smoothed_right.startswith("fingers_") or smoothed_right == "open_palm":
                    count = count_extended_fingers(right_lm)
                else:
                    count = 0  # fist, none → delay off
                subdiv_buf.append(count)
            # pinching or hand lost: do not append; subdivision latches at last value

            if len(subdiv_buf) == SUBDIV_BUF_LEN:
                candidate = mode(subdiv_buf)
                if subdiv_buf.count(candidate) >= SUBDIV_MIN_VOTES:
                    state.delay_subdivision = candidate

            # --- BOTH HANDS PINCH: elastic tightness from inter-hand distance ---
            if smoothed_left == "pinch" and smoothed_right == "pinch" \
                    and left_lm is not None and right_lm is not None:
                # use index fingertips (lm[8]) as the pinch-point reference
                lx, ly = left_lm[8].x, left_lm[8].y
                rx, ry = right_lm[8].x, right_lm[8].y
                dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
                # close together (small dist) → tightness 1.0 (loose/jumpy)
                # far apart (large dist)      → tightness 0.0 (tight/sequential)
                raw = 1.0 - max(0.0, min(1.0,
                    (dist - PINCH_DIST_MIN) / (PINCH_DIST_MAX - PINCH_DIST_MIN)))
                tightness_ema = tightness_ema * (1 - TIGHTNESS_ALPHA) + raw * TIGHTNESS_ALPHA
            # else: not pinching — hold last value, no drift

            state.tightness = _smoothstep(tightness_ema)

            state.left_gesture = smoothed_left
            state.right_gesture = smoothed_right

            # write frame + result for main thread to display
            state.frame = frame
            state.result = result

    finally:
        landmarker.close()
        cap.release()
