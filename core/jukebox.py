import argparse
import time
import numpy as np
import librosa
import scipy.spatial.distance
import sounddevice as sd
from pynput import keyboard


# global state for interactive playback
class JukeboxState:
    def __init__(self):
        self.current_beat = 0
        self.position_in_beat = 0
        self.space_held = False
        self.jump_requested = False
        self.running = True
        self.eighth_note_pos = 0 # for looping eighth notes


def beat_track(music, sr, hop_length):
    # get beat frames using librosa's beat tracker
    _, beats = librosa.beat.beat_track(y=music, sr=sr, hop_length=hop_length)
    return beats

def beat_sync_features(feature_vectors, beats, aggregator=np.median):
    # grab the beat sync features by aggregating chroma features within each beat segment
    
    beat_synced = []

    for i in range(len(beats) - 1):
        start, end = beats[i], beats[i + 1]
        if start < end:
            segment = feature_vectors[:, start:end]
            beat_synced.append(aggregator(segment, axis=1))

    return np.array(beat_synced).T

def similarity_matrix(feature_vectors, distance_metric='cosine'):
    # compute pairwise distances, convert to similarity matrix
    pdist = scipy.spatial.distance.pdist(feature_vectors.T, metric=distance_metric)
    return scipy.spatial.distance.squareform(pdist)

def get_music_samples(music, hop_length, start_frame, end_frame):
    # extract audio samples for a given beat segment
    return music[start_frame * hop_length: end_frame * hop_length]


def analyze_audio(music, sr, hop_length=512):
    # returns beat_chunks (list of audio arrays, one per beat)
    # similarity matrix D
    # timing info for each step (dict, seconds)
    timings = {}
    
    print("Extracting chroma + MFCC features")
    t0 = time.perf_counter()
    chroma = librosa.feature.chroma_cqt(y=music, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=music, sr=sr, hop_length=hop_length, n_mfcc=20)
    timings['chroma'] = time.perf_counter() - t0

    print("Beat tracking")
    t0 = time.perf_counter()
    beats = beat_track(music, sr, hop_length)
    beats = np.append(beats, chroma.shape[1])
    # double the beats by adding midpoints (for now, this is how eight notes are implemented)
    doubled = []
    for i in range(len(beats) - 1):
        doubled.append(beats[i])
        doubled.append((beats[i] + beats[i + 1]) // 2)
    doubled.append(beats[-1])
    beats = np.array(doubled)
    timings['beat_track'] = time.perf_counter() - t0

    print(f"Found {len(beats)} beats (doubled)")

    print("Computing beat-sync features")
    t0 = time.perf_counter()
    beat_synced_chroma = beat_sync_features(chroma, beats, aggregator=np.median)
    beat_synced_mfcc   = beat_sync_features(mfcc,   beats, aggregator=np.median)
    beat_synced_features = np.vstack([beat_synced_chroma, beat_synced_mfcc])  # (32, N)
    timings['beat_sync'] = time.perf_counter() - t0
    
    print("Computing similarity matrix")
    t0 = time.perf_counter()
    D = similarity_matrix(beat_synced_features, distance_metric='cosine')
    timings['similarity'] = time.perf_counter() - t0
    
    print("Extracting beat audio chunks")
    t0 = time.perf_counter()
    beat_chunks = []
    for i in range(len(beats) - 1):
        start_f = int(beats[i])
        end_f = int(beats[i + 1])
        chunk = get_music_samples(music, hop_length, start_f, end_f)
        beat_chunks.append(chunk)
    timings['chunks'] = time.perf_counter() - t0
    
    print("\n--- Timing ---")
    print(f"Chroma extraction: {timings['chroma']*1000}ms")
    print(f"Beat tracking: {timings['beat_track']*1000}ms")
    print(f"Beat sync: {timings['beat_sync']*1000}ms")
    print(f"Similarity matrix: {timings['similarity']*1000}ms")
    print(f"Chunk extraction: {timings['chunks']*1000}ms")
    print(f"Total analysis: {sum(timings.values())*1000}ms\n")
    
    return beat_chunks, D, beat_synced_features, timings

def find_most_similar_beat(D, current_beat):
    distances = D[current_beat].copy()
    distances[current_beat] = np.inf # ignore self
    return int(np.argmin(distances))


def stream_jukebox(beat_chunks, D, sr):

    # main streaming function
    # controls:
        # space (hold): loop current beat
        # f: jump to most similar beat
        # Ctrl C: stop 
    state = JukeboxState()
    num_beats = len(beat_chunks)
    
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                state.space_held = True
            elif hasattr(key, 'char') and key.char == 'f':
                state.jump_requested = True
        except AttributeError:
            pass
    
    def on_release(key):
        if key == keyboard.Key.space:
            state.space_held = False

    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(status)
        
        output_pos = 0
        while output_pos < frames and state.running:
            current_chunk = beat_chunks[state.current_beat]
            chunk_len = len(current_chunk)
            
            if state.space_held:
                # loop current beat
                eighth_len = chunk_len // 8
                if eighth_len > 0:
                    eighth_start = (state.eighth_note_pos % 8) * eighth_len
                    eighth_end = eighth_start + eighth_len
                    sub_chunk = current_chunk[eighth_start:eighth_end]
                    
                    available = len(sub_chunk) - state.position_in_beat
                    to_copy = min(available, frames - output_pos)
                    
                    if to_copy > 0:
                        outdata[output_pos:output_pos + to_copy, 0] = sub_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                        output_pos += to_copy
                        state.position_in_beat += to_copy
                    
                    if state.position_in_beat >= len(sub_chunk):
                        state.position_in_beat = 0
                        state.eighth_note_pos += 1
                else:
                    # beat too short, just loop whole beat
                    available = chunk_len - state.position_in_beat
                    to_copy = min(available, frames - output_pos)
                    outdata[output_pos:output_pos + to_copy, 0] = current_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                    output_pos += to_copy
                    state.position_in_beat += to_copy
                    if state.position_in_beat >= chunk_len:
                        state.position_in_beat = 0
            else:
                # normal playback
                state.eighth_note_pos = 0 # reset eighth note position
                available = chunk_len - state.position_in_beat
                to_copy = min(available, frames - output_pos)
                
                if to_copy > 0:
                    outdata[output_pos:output_pos + to_copy, 0] = current_chunk[state.position_in_beat:state.position_in_beat + to_copy]
                    output_pos += to_copy
                    state.position_in_beat += to_copy
                
                if state.position_in_beat >= chunk_len:
                    # move to next beat
                    state.position_in_beat = 0
                    
                    if state.jump_requested:
                        # jump to most similar beat
                        new_beat = find_most_similar_beat(D, state.current_beat)
                        print(f"Jump: {state.current_beat} -> {new_beat}")
                        state.current_beat = new_beat
                        state.jump_requested = False
                    else:
                        # normal progression
                        state.current_beat = (state.current_beat + 1) % num_beats
        
        # fill any remaining with zeros
        if output_pos < frames:
            outdata[output_pos:, 0] = 0

    print("\n=== Interactive Jukebox ===")
    print("Controls:")
    print(" SPACE (hold): Loop current beat (eighth notes)")
    print(" F: Jump to most similar beat")
    print(" Ctrl+C: Stop\n")
    
    # start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    try:
        with sd.OutputStream(samplerate=sr, channels=1, callback=audio_callback):
            while state.running:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        state.running = False
        listener.stop()

def main():
    parser = argparse.ArgumentParser(description="Interactive Infinite Jukebox")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--hop", type=int, default=512,
                        help="Hop length for analysis (default: 512)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    print(f"Loading {args.audio_file}")
    t0 = time.perf_counter()
    music, sr = librosa.load(args.audio_file)
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(music) / sr:.1f}s of audio at {sr}Hz ({load_time*1000:.1f}ms)")
    
    beat_chunks, D, _, _ = analyze_audio(music, sr, hop_length=args.hop)
    
    stream_jukebox(beat_chunks, D, sr)


if __name__ == "__main__":
    main()
