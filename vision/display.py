import cv2
from core.handsoff import CONNECTIONS, dist, is_pinch, is_fist, is_open_palm


def draw_frame(frame, result, state):
    """Draw hand skeleton, per-hand gesture labels, and beat counter onto frame."""
    h, w = frame.shape[:2]

    if result and result.hand_landmarks:
        for idx, (lm, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            # draw skeleton (landmarks are from unflipped detection, mirror x for display)
            for a, b in CONNECTIONS:
                p1 = (int((1 - lm[a].x) * w), int(lm[a].y * h))
                p2 = (int((1 - lm[b].x) * w), int(lm[b].y * h))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            # classify gesture for label
            if is_pinch(lm):
                gesture_label = "PINCH"
                color = (255, 0, 255)
            elif is_fist(lm):
                gesture_label = "FIST"
                color = (0, 128, 255)
            elif is_open_palm(lm):
                gesture_label = "OPEN PALM"
                color = (0, 255, 0)
            else:
                gesture_label = "..."
                color = (200, 200, 200)

            mp_hand = handedness[0].category_name
            user_hand = "LEFT" if mp_hand == "Left" else "RIGHT"

            # place label on left side of frame for user's left hand, right side for right
            if user_hand == "LEFT":
                x_pos = 10
            else:
                x_pos = w - 220

            y_pos = 50 + idx * 60
            cv2.putText(frame, f"{user_hand}: {gesture_label}", (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # param readout below gesture label
            if user_hand == "LEFT":
                param_label = f"LPF: {int(state.lpf_cutoff * 100)}%"
                cv2.putText(frame, param_label, (10, y_pos + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            else:
                param_label = f"Tight: {int(state.tightness * 100)}%"
                cv2.putText(frame, param_label, (w - 220, y_pos + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

    # active clusters + pending selection
    active_str = "Clusters: " + ", ".join(str(c + 1) for c in sorted(state.active_clusters))
    cv2.putText(frame, active_str, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
    if state.cluster_pending > 0:
        pct = min(state.cluster_pending_t / 2.0, 1.0)
        hold_str = f"Hold {state.cluster_pending}: {state.cluster_pending_t:.1f}s"
        cv2.putText(frame, hold_str, (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2)
        # progress bar
        bar_w = int((w // 3) * pct)
        cv2.rectangle(frame, (10, h - 58), (10 + w // 3, h - 50), (80, 80, 80), -1)
        cv2.rectangle(frame, (10, h - 58), (10 + bar_w, h - 50), (255, 255, 100), -1)

    # beat counter in bottom-left
    cv2.putText(frame, f"Beat: {state.current_beat}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
