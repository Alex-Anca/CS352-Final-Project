from random import random

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

from core.state import SharedState


_XFADE_LEN = 256  # ~11ms at 22050 Hz


def make_audio_callback(beat_chunks, D, cluster_labels, state: SharedState, sr: int = 22050):
    """Return a sounddevice callback closure that reads from SharedState."""
    num_beats = len(beat_chunks)
    _sr = sr  # capture for LPF frequency calculation

    # LPF state - redesigned lazily when cutoff changes meaningfully
    lpf_sos = butter(2, 0.99, btype='low', output='sos')
    lpf_zi = sosfilt_zi(lpf_sos) * 0.0
    last_lpf_cutoff = 1.0

    def audio_callback(outdata, frames, time_info, status):
        nonlocal lpf_sos, lpf_zi, last_lpf_cutoff

        if status:
            print(status)

        output_pos = 0

        while output_pos < frames and state.running:
            current_chunk = beat_chunks[state.current_beat]
            chunk_len = len(current_chunk)

            left = state.left_gesture
            right = state.right_gesture

            # --- LEFT FIST: loop full beat (highest priority loop) ---
            if left == "fist":
                available = chunk_len - state.position_in_beat
                to_copy = min(available, frames - output_pos)
                if to_copy > 0:
                    outdata[output_pos:output_pos + to_copy, 0] = \
                        current_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                    output_pos += to_copy
                    state.position_in_beat += to_copy
                if state.position_in_beat >= chunk_len:
                    state.position_in_beat = 0

            # --- RIGHT FIST: eighth-note stutter ---
            elif right == "fist":
                eighth_len = chunk_len // 8
                if eighth_len > 0:
                    eighth_start = (state.eighth_note_pos % 8) * eighth_len
                    sub_chunk = current_chunk[eighth_start:eighth_start + eighth_len]
                    available = len(sub_chunk) - state.position_in_beat
                    to_copy = min(available, frames - output_pos)
                    if to_copy > 0:
                        outdata[output_pos:output_pos + to_copy, 0] = \
                            sub_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                        output_pos += to_copy
                        state.position_in_beat += to_copy
                    if state.position_in_beat >= len(sub_chunk):
                        state.position_in_beat = 0
                        state.eighth_note_pos += 1
                else:
                    # beat too short - loop whole beat
                    available = chunk_len - state.position_in_beat
                    to_copy = min(available, frames - output_pos)
                    outdata[output_pos:output_pos + to_copy, 0] = \
                        current_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                    output_pos += to_copy
                    state.position_in_beat += to_copy
                    if state.position_in_beat >= chunk_len:
                        state.position_in_beat = 0

            # --- NORMAL PLAYBACK ---
            else:
                state.eighth_note_pos = 0
                available = chunk_len - state.position_in_beat
                to_copy = min(available, frames - output_pos)
                if to_copy > 0:
                    outdata[output_pos:output_pos + to_copy, 0] = \
                        current_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                    output_pos += to_copy
                    state.position_in_beat += to_copy

                if state.position_in_beat >= chunk_len:
                    state.position_in_beat = 0

                    active = state.active_clusters
                    if cluster_labels is not None and len(active) > 0:
                        all_active = np.where(np.isin(cluster_labels, list(active)))[0]
                        pool = all_active[all_active != state.current_beat]
                    else:
                        all_active = np.array([], dtype=np.int64)
                        pool = np.array([], dtype=np.int64)

                    tightness = state.tightness
                    next_seq = (state.current_beat + 1) % num_beats

                    if len(pool) > 0 and random() < tightness:
                        new_beat = int(np.random.choice(pool))
                        is_jump = True
                    elif tightness > 0 and cluster_labels is not None and len(all_active) > 0 and cluster_labels[next_seq] not in active:
                        ahead = all_active[all_active > state.current_beat]
                        new_beat = int(ahead[0]) if len(ahead) > 0 else int(all_active[0])
                        is_jump = True
                    else:
                        new_beat = next_seq
                        is_jump = False

                    state.prev_beat = state.current_beat
                    state.current_beat = new_beat

                    # crossfade only for non-sequential jumps
                    if is_jump:
                        xn = min(_XFADE_LEN, output_pos, len(beat_chunks[new_beat]))
                        if xn > 0:
                            t = np.linspace(0, 1, xn, dtype=np.float32)
                            outdata[output_pos - xn:output_pos, 0] *= (1 - t)
                            outdata[output_pos - xn:output_pos, 0] += beat_chunks[new_beat][:xn] * t
                            state.position_in_beat = xn

        if output_pos < frames:
            outdata[output_pos:, 0] = 0

        # --- IIR LOW-PASS FILTER ---
        cutoff = state.lpf_cutoff
        if cutoff < 0.999:
            if abs(cutoff - last_lpf_cutoff) > 0.005:
                freq = min(100.0 * (20000.0 / 100.0) ** cutoff, _sr * 0.49)
                lpf_sos = butter(2, freq / (_sr / 2), btype='low', output='sos')
                if last_lpf_cutoff >= 0.999:
                    # transitioning from bypass: seed zi to avoid onset click
                    lpf_zi = sosfilt_zi(lpf_sos) * outdata[0, 0]
                # else: filter already running - carry zi over, no reseed
                last_lpf_cutoff = cutoff
            filtered, lpf_zi = sosfilt(lpf_sos, outdata[:frames, 0], zi=lpf_zi)
            outdata[:frames, 0] = filtered
        else:
            last_lpf_cutoff = cutoff

    return audio_callback


def run_audio_stream(beat_chunks, D, cluster_labels, sr, state: SharedState) -> None:
    """Open sounddevice output stream and block until state.running is False."""
    callback = make_audio_callback(beat_chunks, D, cluster_labels, state, sr=sr)
    with sd.OutputStream(samplerate=sr, channels=1, callback=callback):
        while state.running:
            sd.sleep(100)
