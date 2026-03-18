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
| `core/handsoff.py` | Gesture classifiers: `is_open_palm`, `is_fist`, `is_pinch`, `count_extended_fingers`; drawing constants | mediapipe, numpy |
| `vision/gesture.py` | Camera loop, MediaPipe inference, EMA smoothing, gesture classification → writes to state | core.handsoff, core.state |
| `audio/player.py` | sounddevice callback, beat navigation, pedalboard effects chain → reads from state | core.jukebox, core.state |
| `vision/display.py` | OpenCV overlay: hand skeleton, SPACE/TIME/TENSION panels, rubber-band line | core.handsoff, cv2 |
| `viz/timeline.py` | Circular beat timeline: notches colored by similarity, jump arcs, active beat highlight | numpy, cv2 |
| `main.py` | Argument parsing, audio loading, thread orchestration | all of the above |

---

## Adding a New Gesture

1. **`core/handsoff.py`** — add a classifier function `is_<gesture>(lm) -> bool`
2. **`vision/gesture.py`** — add a branch in `classify_gesture()` returning the new gesture string
3. **`core/state.py`** — add a new field if the gesture needs to communicate a value to the audio thread
4. **`audio/player.py`** — read the new state field inside `make_audio_callback`

No other files need to change.

---

## Adding a New Audio Effect

1. **`core/state.py`** — add a state field (e.g., `pitch_shift: float = 0.0`)
2. **`audio/player.py`** — add the effect to the pedalboard chain and gate it on the state field
3. **`vision/gesture.py`** — write the field when the appropriate gesture is detected

---

## Handedness Convention

MediaPipe reports handedness from the camera's perspective (unflipped image).
Since the display is mirrored, the labels are flipped relative to the user:

- MediaPipe "Left"  → user's **right** hand
- MediaPipe "Right" → user's **left** hand

This flip is applied in `vision/gesture.py` and `vision/display.py`. Keep this in mind when
assigning gestures to hands.

---

## Gesture Smoothing

Discrete gestures (gesture label per hand) use a `deque(maxlen=4)` rolling buffer.
The smoothed label is the `statistics.mode` of the last 4 frames — majority rule, no minimum
vote count. This filters single-frame noise with minimal latency (~4 frames ≈ 130ms at 30fps).

The delay subdivision uses a stricter separate buffer: `deque(maxlen=8)` requiring at least
5 matching votes before the subdivision value commits, to prevent subdivision from flickering
during finger transitions.

---

## Continuous Parameter Smoothing

All continuous parameters use Exponential Moving Average (EMA):

```
ema = ema * (1 - alpha) + raw * alpha
```

| Parameter | State field | EMA alpha | Source |
|-----------|-------------|-----------|--------|
| Reverb depth | `reverb_depth` | 0.20 | Left hand wrist-to-knuckle distance |
| Reverb tail | `reverb_tail` | 0.12 | Left hand extended finger count / 4 |
| Delay feedback | `delay_depth` | 0.20 | Right hand wrist-to-knuckle distance |
| Tension | `tightness` | 0.08 | Pinch inter-hand distance (inverted) |

All parameters are passed through a smoothstep function before being written to state,
which causes them to snap cleanly to 0 or 1 near the extremes.

All parameters hold their last value when the hand is not detected (no decay).

---

## Audio Effects Chain

Effects are processed via Spotify's **pedalboard** library inside the sounddevice callback:

```
dry audio → Reverb → Delay → output
```

- **Reverb**: `wet_level = reverb_depth`, `room_size` driven by `reverb_tail`
- **Delay**: `delay_seconds` set by subdivision × beat duration, `feedback` driven by `delay_depth`
- When delay subdivision is 0 (off), `delay.mix = 0.0`

The pedalboard is called with `reset=False` so the reverb tail and delay repeats persist
across beat boundaries.

---

## Beat Navigation

Each beat, the player decides whether to jump or continue sequentially based on `state.tightness`:

```python
if random() < tightness:
    # jump to most similar beat (argmin of similarity matrix row)
else:
    # advance to next beat sequentially
```

The similarity matrix `D` holds cosine distances between beat feature vectors
(chroma + MFCC). Lower = more similar.

---

## Thread Model

```
main thread
  ├── gesture thread (daemon)   — camera + MediaPipe + EMA smoothing → writes state
  └── audio thread (daemon)     — sounddevice callback + pedalboard effects → reads state
```

The GIL makes simple attribute reads/writes on the shared `SharedState` dataclass safe
across threads without explicit locks.
