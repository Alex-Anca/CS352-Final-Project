import numpy as np
import cv2
from core.handsoff import CONNECTIONS, is_fist, is_open_palm, is_pinch

FONT = cv2.FONT_HERSHEY_DUPLEX

SUBDIV_LABELS = {0: "Off", 1: "1/4", 2: "1/8", 3: "3/8", 4: "1/16"}


def draw_frame(frame, result, state):
    """Draw instrument panels, hand skeletons, and gesture labels onto frame."""
    h, w = frame.shape[:2]

    # --- Hand skeletons + gesture labels (fixed position per side) ---
    if result and result.hand_landmarks:
        for lm, handedness in zip(result.hand_landmarks, result.handedness):
            for a, b in CONNECTIONS:
                p1 = (int((1 - lm[a].x) * w), int(lm[a].y * h))
                p2 = (int((1 - lm[b].x) * w), int(lm[b].y * h))
                cv2.line(frame, p1, p2, (0, 200, 0), 2)

            if is_fist(lm):
                gesture_label, color = "FIST", (0, 140, 255)
            elif is_pinch(lm):
                gesture_label, color = "PINCH", (120, 255, 160)
            elif is_open_palm(lm):
                gesture_label, color = "OPEN", (0, 230, 0)
            else:
                gesture_label, color = "...", (140, 140, 140)

            mp_hand = handedness[0].category_name
            user_hand = "LEFT" if mp_hand == "Left" else "RIGHT"
            x_pos = 10 if user_hand == "LEFT" else w - 220

            # fixed y=130 regardless of detection order — clears both panels
            cv2.putText(frame, gesture_label, (x_pos, 130),
                        FONT, 0.6, color, 1)

    # --- Rubber band between index fingertips when both hands are pinching ---
    if (state.left_gesture == "pinch" and state.right_gesture == "pinch"
            and result and result.hand_landmarks):
        left_tip = right_tip = None
        for lm, handedness in zip(result.hand_landmarks, result.handedness):
            tip = (int((1 - lm[8].x) * w), int(lm[8].y * h))
            if handedness[0].category_name == "Left":
                left_tip = tip
            else:
                right_tip = tip

        if left_tip and right_tip:
            tightness = state.tightness       # 1.0 = close = loose band
            tautness  = 1.0 - tightness       # 1.0 = far   = taut band

            # Control point sags down when loose, sits at midpoint when taut
            mid_x = (left_tip[0] + right_tip[0]) // 2
            mid_y = (left_tip[1] + right_tip[1]) // 2
            control = (mid_x, mid_y + int(tightness * 55))

            # Mint-green (loose) → amber (taut)
            color = (80, int(230 - tautness * 80), int(80 + tautness * 170))
            thickness = max(1, int(3 - tautness * 2))

            # Quadratic bezier via polylines
            steps = 30
            pts = np.array(
                [[[int((1-t)**2 * left_tip[0] + 2*(1-t)*t * control[0] + t**2 * right_tip[0]),
                   int((1-t)**2 * left_tip[1] + 2*(1-t)*t * control[1] + t**2 * right_tip[1])]]
                 for t in (i / steps for i in range(steps + 1))],
                dtype=np.int32,
            )

            # Blend for variable opacity: translucent when loose, nearly opaque when taut
            alpha = 0.35 + tautness * 0.55
            overlay = frame.copy()
            cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # --- SPACE panel (left side, always visible) ---
    bar_w = 160
    cv2.putText(frame, "SPACE", (10, 22), FONT, 0.45, (140, 100, 200), 1)
    # depth (wet)
    fill = int(state.reverb_depth * bar_w)
    cv2.putText(frame, f"Wet  {int(state.reverb_depth * 100)}%", (10, 44), FONT, 0.45, (180, 120, 255), 1)
    cv2.rectangle(frame, (10, 49), (10 + bar_w, 61), (55, 55, 55), -1)
    cv2.rectangle(frame, (10, 49), (10 + fill, 61), (180, 120, 255), -1)
    # tail (room size)
    tail_fill = int(state.reverb_tail * bar_w)
    cv2.putText(frame, f"Tail  {int(state.reverb_tail * 100)}%", (10, 80), FONT, 0.45, (200, 150, 255), 1)
    cv2.rectangle(frame, (10, 85), (10 + bar_w, 97), (55, 55, 55), -1)
    cv2.rectangle(frame, (10, 85), (10 + tail_fill, 97), (200, 150, 255), -1)

    # --- TIME panel (right side, always visible) ---
    subdiv_label = SUBDIV_LABELS.get(state.delay_subdivision, "?")
    fb_pct = int(state.delay_depth * 100)
    fb_bar_w = 160
    fb_fill = int(state.delay_depth * fb_bar_w)
    cv2.putText(frame, "TIME", (w - 220, 26), FONT, 0.5, (70, 170, 200), 1)
    cv2.putText(frame, subdiv_label, (w - 220, 58), FONT, 0.85, (100, 220, 255), 2)
    cv2.putText(frame, f"FB  {fb_pct}%", (w - 220, 85), FONT, 0.6, (100, 210, 210), 1)
    cv2.rectangle(frame, (w - 220, 92), (w - 220 + fb_bar_w, 106), (55, 55, 55), -1)
    cv2.rectangle(frame, (w - 220, 92), (w - 220 + fb_fill, 106), (80, 210, 210), -1)

    # --- TENSION bar (bottom centre, always visible) ---
    bar_w2 = 200
    fill2 = int(state.tightness * bar_w2)
    bx = w // 2 - bar_w2 // 2
    by = h - 50
    active = (state.left_gesture == "pinch" and state.right_gesture == "pinch")
    bar_col   = (120, 255, 160) if active else (70, 140, 90)
    label_col = (160, 255, 180) if active else (90, 150, 105)
    cv2.putText(frame, "TENSION", (bx + bar_w2 // 2 - 38, by - 8),
                FONT, 0.45, label_col, 1)
    cv2.putText(frame, "TIGHT", (bx - 50, by + 13), FONT, 0.38, label_col, 1)
    cv2.putText(frame, "LOOSE", (bx + bar_w2 + 4, by + 13), FONT, 0.38, label_col, 1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w2, by + 14), (45, 45, 45), -1)
    cv2.rectangle(frame, (bx, by), (bx + fill2, by + 14), bar_col, -1)

    # --- Beat counter ---
    cv2.putText(frame, f"Beat  {state.current_beat}", (10, h - 12),
                FONT, 0.45, (160, 160, 160), 1)
