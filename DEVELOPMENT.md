# Development Guide

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe hand model:
```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Run:
```bash
python main.py media/[mediafile.mp3]
```

---

## Module Responsibilities

| File | Role | Imports from |
|------|------|--------------|
| `core/state.py` | Shared data between threads (pure dataclass, no logic) | nothing |
| `core/jukebox.py` | Audio analysis: chroma, beat tracking, similarity matrix, chunk extraction | librosa, scipy |
| `core/handsoff.py` | Gesture classifiers: `is_open_palm`, `is_fist`, `is_pinch`; drawing constants | mediapipe, numpy |
| `vision/gesture.py` | Camera loop, MediaPipe inference, smoothing buffer, edge detection → writes to state | core.handsoff, core.state, vision.display |
| `audio/player.py` | sounddevice callback, beat navigation logic → reads from state | core.jukebox, core.state |
| `vision/display.py` | OpenCV overlay: skeleton, gesture labels, beat counter | core.handsoff, cv2 |
| `main.py` | Argument parsing, audio loading, thread orchestration | all of the above |

---

## Adding a New Gesture

1. **`core/handsoff.py`** - add a classifier function `is_<gesture>(lm) -> bool`
2. **`vision/gesture.py`** - add a branch in `classify_gesture()` returning the new gesture string
3. **`core/state.py`** - add a new flag if the gesture needs edge-triggered behavior (e.g., `jump_new: bool = False`)
4. **`audio/player.py`** - add a case in `make_audio_callback` that reads the new state flag and performs the audio action

No other files need to change.

---

## Adding a New Audio Effect

1. **`core/state.py`** - add a state flag (e.g., `reverse_mode: bool = False`)
2. **`audio/player.py`** - add the effect logic inside `make_audio_callback`, gated on the flag
3. **`vision/gesture.py`** - set the flag when the appropriate gesture is detected

---

## Handedness Convention

MediaPipe reports handedness from the camera's perspective (unflipped image).
Since the display is mirrored, the labels are flipped relative to the user:

- MediaPipe "Left"  → user's **right** hand
- MediaPipe "Right" → user's **left** hand

This flip is applied in `vision/gesture.py` and `vision/display.py`. Keep this in mind when
assigning gestures to hands.

---

## Smoothing

Each hand has a `deque(maxlen=4)` rolling buffer. The smoothed gesture is
`statistics.mode(buffer)`. This filters single-frame noise without introducing
noticeable latency (~4 frames ≈ 130ms at 30fps).

---

## Thread Model

```
main thread
  ├── gesture thread (daemon)   - camera + MediaPipe + writes state
  └── audio thread (main)       - sounddevice callback + reads state
```

The GIL makes simple attribute reads/writes safe across threads without locks.
`jump_left` and `jump_right` are boolean flags consumed (set to False) by the
audio callback after acting on them - this is safe because only one thread
writes True (gesture) and only one thread resets to False (audio).
