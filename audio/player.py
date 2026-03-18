from random import random

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi
from pedalboard import Pedalboard, Reverb, Delay

from core.state import SharedState


_XFADE_LEN = 256  # ~11ms at 22050 Hz

# ── Effect Config, tunable
AUTO_TIGHTNESS      = 0.15   # fixed jump probability (replaces gesture tightness)
REVERB_ROOM_MIN     = 0.30
REVERB_ROOM_MAX     = 0.90
REVERB_DAMP_MAX     = 0.50
DELAY_MIX           = 0.40
DELAY_FEEDBACK_BASE = 0.30
DELAY_FEEDBACK_MAX  = 0.55
# subdivisions as beat multipliers (quarter-note = 1.0 beat)
DELAY_SUBDIV_BEATS  = {1: 1.0, 2: 0.5, 3: 0.75, 4: 0.25}


def make_audio_callback(beat_chunks, D, cluster_labels, state: SharedState,
                        sr: int = 22050, bpm: float = 120.0):
    """Return a sounddevice callback closure that reads from SharedState."""
    num_beats = len(beat_chunks)
    _sr = sr  # capture for LPF frequency calculation

    # LPF state - redesigned lazily when cutoff changes meaningfully
    lpf_sos = butter(2, 0.99, btype='low', output='sos')
    lpf_zi = sosfilt_zi(lpf_sos) * 0.0
    last_lpf_cutoff = 1.0

    # Pedalboard effects (stateful: carry internal buffers across callbacks)
    _reverb = Reverb(room_size=0.3, damping=0.0, wet_level=0.0, dry_level=1.0)
    _delay  = Delay(delay_seconds=0.5, feedback=0.30, mix=0.0)
    _board  = Pedalboard([_reverb, _delay])

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

            # --- RIGHT FIST: loop full beat ---
            if right == "fist":
                available = chunk_len - state.position_in_beat
                to_copy = min(available, frames - output_pos)
                if to_copy > 0:
                    outdata[output_pos:output_pos + to_copy, 0] = \
                        current_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                    output_pos += to_copy
                    state.position_in_beat += to_copy
                if state.position_in_beat >= chunk_len:
                    state.position_in_beat = 0

            # --- NORMAL PLAYBACK ---
            else:
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

                    if random() < tightness:
                        # jump to the most similar beat (min distance in D, excluding self)
                        row = D[state.current_beat].copy()
                        row[state.current_beat] = np.inf
                        new_beat = int(np.argmin(row))
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

        # --- LPF auto-coupled to reverb depth (deeper = slightly darker) ---
        state.lpf_cutoff = max(0.0, min(1.0, 1.0 - state.reverb_depth * 0.35))

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

        # --- UPDATE REVERB + DELAY, THEN PROCESS ---
        if frames > 0:
            depth  = state.reverb_depth
            subdiv = state.delay_subdivision

            _reverb.wet_level = depth
            _reverb.dry_level = 1.0 - depth * 0.9   # at depth=1.0, dry=0.1 (fully drowned)
            _reverb.room_size = REVERB_ROOM_MIN + state.reverb_tail * (REVERB_ROOM_MAX - REVERB_ROOM_MIN)
            _reverb.damping   = depth * REVERB_DAMP_MAX

            if subdiv == 0:
                _delay.mix = 0.0
            else:
                beat_sec = 60.0 / bpm
                _delay.delay_seconds = beat_sec * DELAY_SUBDIV_BEATS[subdiv]
                _delay.feedback = min(DELAY_FEEDBACK_BASE + state.delay_depth * 0.25, DELAY_FEEDBACK_MAX)
                _delay.mix = DELAY_MIX

            # pedalboard expects (channels, samples); we're mono
            dry = outdata[:frames, 0].copy().reshape(1, frames)
            wet = _board(dry, sample_rate=_sr, reset=False)
            outdata[:frames, 0] = wet[0]

    return audio_callback


def run_audio_stream(beat_chunks, D, cluster_labels, sr, state: SharedState,
                     bpm: float = 120.0) -> None:
    """Open sounddevice output stream and block until state.running is False."""
    callback = make_audio_callback(beat_chunks, D, cluster_labels, state,
                                   sr=sr, bpm=bpm)
    with sd.OutputStream(samplerate=sr, channels=1, callback=callback):
        while state.running:
            sd.sleep(100)
