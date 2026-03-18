#!/usr/bin/env python3
"""
Gesture-Controlled Infinite Jukebox
------------------------------------
main.py - entry point, wires gesture + audio threads together.

Usage:
    python main.py media/[mediafile.mp3]
    python main.py media/[mediafile.mp3] --hop 512 --seed 42
"""

import os
# Redirect fd 2 (stderr) to /dev/null - silences C++ library noise from
# MediaPipe, TFLite, and OpenCV/Qt. All user-facing output uses print() -> stdout.
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
os.close(_devnull)

import argparse
import threading
import time

import cv2
import librosa
import numpy as np
from pynput import keyboard

from core.jukebox import analyze_audio
from core.state import SharedState
from vision.gesture import run_gesture_thread
from vision.display import draw_frame
from audio.player import run_audio_stream
from viz.timeline import TimelineRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-Controlled Infinite Jukebox")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--hop", type=int, default=512,
                        help="Hop length for analysis (default: 512)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading {args.audio_file}")
    t0 = time.perf_counter()
    music, sr = librosa.load(args.audio_file)
    print(f"Loaded {len(music) / sr:.1f}s of audio at {sr}Hz ({(time.perf_counter()-t0)*1000:.1f}ms)")

    beat_chunks, D, beat_synced_features, _, bpm = analyze_audio(music, sr, hop_length=args.hop)

    beat_chunk_lengths = [len(c) for c in beat_chunks]
    renderer = TimelineRenderer(len(beat_chunks), D, beat_chunk_lengths, canvas_size=(720, 720))

    state = SharedState()

    print("\n=== Gesture Controls ===")
    print(f" BPM detected: {bpm:.1f}")
    print(" LEFT hand forward/back  : reverb depth (closer = sound source farther back)")
    print(" LEFT hand finger spread : reverb tail  (open = big room, fist = small room)")
    print(" RIGHT hand fingers      : delay subdivision  1=1/4  2=1/8  3=3/8  4=1/16")
    print(" RIGHT hand forward/back : delay feedback (closer = more repeats)")
    print(" RIGHT fist / open palm  : delay off")
    print(" BOTH hands pinch        : TENSION  (far apart = sequential, close = jump often)")
    print(" Q / Ctrl+C              : Quit\n")

    def on_press(key):
        if hasattr(key, 'char') and key.char == 'q':
            state.running = False
            return False  # stop listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    gesture_t = threading.Thread(target=run_gesture_thread, args=(state,))
    gesture_t.start()

    audio_t = threading.Thread(target=run_audio_stream,
                               args=(beat_chunks, D, None, sr, state),
                               kwargs={"bpm": bpm})
    audio_t.start()

    # Display loop runs on the main thread - required by Qt/OpenCV
    try:
        while state.running:
            frame = state.frame
            result = state.result
            if frame is not None:
                display = cv2.flip(frame, 1)
                draw_frame(display, result, state)

                canvas = renderer.render(state)
                display = cv2.resize(display, (960, 720), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display, canvas])
                cv2.imshow('Gesture Jukebox', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                state.running = False
                break
    except KeyboardInterrupt:
        state.running = False
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("\nStopped.")

    listener.stop()
    audio_t.join(timeout=3.0)
    gesture_t.join(timeout=3.0)


if __name__ == "__main__":
    main()
