# Hands Off — Gesture-Controlled Spatial Music Instrument

A song is analyzed into beats and played back as an infinite loop. Your two hands each control a different dimension of sound in real time via webcam.

## Demo

![Demo](media/Final%20Project%20335.mp4)

*Live performance with Hands Off — "Bees" by Animal Collective*

---

## How to Run

```bash
source venv/bin/activate
python main.py media/[mediafile.mp3]
```

**Optional flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--hop N` | Hop length for audio analysis | 512 |
| `--seed N` | Random seed for reproducibility | — |

Press `Q` or `Ctrl+C` to quit.

---

## Gesture Controls

![Control Structure](media/Hands%20Off%20Control%20Structure.png)

### Left Hand — SPACE (reverb)

| Gesture | Effect |
|---------|--------|
| Move hand closer to camera | Increase reverb depth (wetter / more distant) |
| Move hand farther from camera | Decrease reverb depth (drier / more present) |
| Spread fingers wider | Longer reverb tail (bigger room) |
| Close fingers toward fist | Shorter reverb tail (tighter room) |

### Right Hand — TIME (delay)

| Gesture | Effect |
|---------|--------|
| 1 finger | Quarter-note delay |
| 2 fingers | Eighth-note delay |
| 3 fingers | Dotted-eighth delay |
| 4 fingers | Sixteenth-note delay |
| Fist | Delay off + loops current beat |
| Move hand closer to camera | Increase delay feedback (more repeats) |
| Move hand farther from camera | Decrease delay feedback (fewer repeats) |

### Both Hands — TENSION

Pinch with both hands and spread or close them to control how the jukebox navigates the song. The gesture acts like an elastic band:

| Position | Effect |
|----------|--------|
| Hands far apart | **Tight** — song plays beat by beat, sequentially |
| Hands close together | **Loose** — jukebox jumps to the most similar-sounding beat |

Tension locks in place when you release the pinch and snaps smoothly to fully tight or fully loose near the extremes.

All parameters hold their last value when a hand is not detected.

---

## Display

On the right side of the screen, a circular beat timeline shows every beat in the song arranged around a loop. Each notch represents one beat, colored by similarity to its neighbors (pink = more similar, purple = less similar). Fading bezier arcs inside the circle trace recent non-sequential jumps, and a green arc points from the current beat to its most similar target. The currently playing beat is highlighted. When both hands are pinching, a rubber-band line connects the two index fingers, growing tauter as the hands move apart.

Inspired by the [Eternal Jukebox](https://eternalbox.floriegl.tech/jukebox_index.html) GUI by floriegl.

---

## Testing

Gesture detection is validated against the [Jester dataset](https://20bn.com/datasets/jester) — 5,400+ short webcam videos spanning 27 hand gesture categories.

```bash
python -m unittest tests.test_gestures -v
```

**Coverage:** 21 tests across `is_open_palm`, `is_fist`, `is_pinch`, and `count_extended_fingers`. Each test aggregates results across 25 randomly sampled clips × 5 middle frames per clip (~100–120 real detected frames per label), giving statistical coverage rather than cherry-picked clip checks. Both positive cases (gesture should fire) and negative cases (gesture should not fire) are tested for each function.

All 21 tests pass in ~30 seconds. Thresholds are conservative because Jester frames are small (100×176 px) and not always captured from ideal angles.

---

## Dependencies

```bash
pip install -r requirements.txt
```

MediaPipe model file required at `models/hand_landmarker.task`:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Sources

Fiebrink, Rebecca, and Perry Cook. "The Wekinator: A System for Real-time, Interactive Machine Learning in Music." In *Proceedings of the Eleventh International Society for Music Information Retrieval Conference (ISMIR 2010)*, 2010.

*Musical Instruments in the 21st Century: Identities, Configurations, Practices.* Singapore: Springer Nature Singapore, 2018.
