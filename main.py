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
from viz.cluster import build_cluster_data, ClusterRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-Controlled Infinite Jukebox")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--hop", type=int, default=512,
                        help="Hop length for analysis (default: 512)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of beat clusters (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading {args.audio_file}")
    t0 = time.perf_counter()
    music, sr = librosa.load(args.audio_file)
    print(f"Loaded {len(music) / sr:.1f}s of audio at {sr}Hz ({(time.perf_counter()-t0)*1000:.1f}ms)")

    beat_chunks, D, beat_synced_features, _ = analyze_audio(music, sr, hop_length=args.hop)

    cluster_labels, pca_coords, k_colors = build_cluster_data(beat_synced_features, k=args.k)
    beat_chunk_lengths = [len(c) for c in beat_chunks]
    # cluster panel is square, same height as camera (determined after first frame)
    # use a fixed size here; main loop will lazy-reinit if camera size differs
    renderer = ClusterRenderer(pca_coords, cluster_labels, k_colors, beat_chunk_lengths,
                               canvas_size=(480, 480))

    state = SharedState()
    state.cluster_labels = cluster_labels
    state.pca_coords = pca_coords

    print("\n=== Gesture Controls ===")
    print(" Left fist   (hold)  : Loop current beat")
    print(" Left pinch  (hold)  : Rotate hand to adjust LPF cutoff (shown as LPF %)")
    print(" Right fist  (hold)  : Eighth-note stutter")
    print(" Right pinch (hold)  : Rotate hand to adjust tightness (shown as Tight %)")
    print(" Q                   : Quit")
    print(" Ctrl+C              : Quit\n")

    def on_press(key):
        if hasattr(key, 'char') and key.char == 'q':
            state.running = False
            return False  # stop listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    gesture_t = threading.Thread(target=run_gesture_thread, args=(state,))
    gesture_t.start()

    audio_t = threading.Thread(target=run_audio_stream, args=(beat_chunks, D, cluster_labels, sr, state))
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
