#!/usr/bin/env python3
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def dist(a, b): # euclidian distance in 3d
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def is_open_palm(lm):
    wrist = lm[0]
    return all([
        dist(lm[4], lm[5]) > dist(lm[3], lm[5]), # thumb extended
        dist(lm[8], wrist)  > dist(lm[6], wrist), # index extended
        dist(lm[12], wrist) > dist(lm[10], wrist), # middle extended
        dist(lm[16], wrist) > dist(lm[14], wrist), # ring extended
        dist(lm[20], wrist) > dist(lm[18], wrist), # pinky extended
    ])

def is_fist(lm):
    wrist = lm[0]
    return all([
        dist(lm[8], wrist)  < dist(lm[6], wrist), # index curled
        dist(lm[12], wrist) < dist(lm[10], wrist), # middle curled
        dist(lm[16], wrist) < dist(lm[14], wrist), # ring curled
        dist(lm[20], wrist) < dist(lm[18], wrist), # pinky curled
    ])

def count_extended_fingers(lm):
    """Return number of extended fingers (0-4, thumb excluded)."""
    wrist = lm[0]
    checks = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(dist(lm[tip], wrist) > dist(lm[pip], wrist) for tip, pip in checks)

def is_pinch(lm):
    # thumb tip to index tip distance relative to hand size
    hand_size = dist(lm[0], lm[9]) # wrist to middle base
    thumb_index_dist = dist(lm[4], lm[8])
    # distinguish from fist: at least one other finger should be extended
    wrist = lm[0]
    other_fingers_extended = any([
        dist(lm[12], wrist) > dist(lm[10], wrist),  # middle
        dist(lm[16], wrist) > dist(lm[14], wrist),  # ring
        dist(lm[20], wrist) > dist(lm[18], wrist),  # pinky
    ])
    return thumb_index_dist < hand_size * 0.3 and other_fingers_extended

def draw(frame, result):
    h, w = frame.shape[:2]
    for lm in result.hand_landmarks:
        for a, b in CONNECTIONS:
            # landmarks are from unflipped detection, mirror x for display
            p1 = (int((1 - lm[a].x) * w), int(lm[a].y * h))
            p2 = (int((1 - lm[b].x) * w), int(lm[b].y * h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
        
        # detect and display gestures 
        # prioritize pinch, as it currently triggers when a fist is made.
        if is_pinch(lm):
            cv2.putText(frame, "PINCH", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
        elif is_fist(lm):
            cv2.putText(frame, "FIST", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 2)
        elif is_open_palm(lm):
            cv2.putText(frame, "OPEN PALM", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

def main():
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    start = time.time()
    # print("Press 'q' to quit") # using ctrl c for now

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            display = cv2.flip(frame, 1)
            if result.hand_landmarks:
                draw(display, result)

            cv2.imshow('handsoff', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
