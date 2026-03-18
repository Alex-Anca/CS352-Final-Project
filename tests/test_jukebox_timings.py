import unittest
import time
import numpy as np
import librosa
from core import jukebox

class TestJukeboxTimings(unittest.TestCase):
    def setUp(self):
        # Generate a short synthetic audio signal (1s of white noise)
        self.sr = 22050
        self.audio = np.random.randn(self.sr)

    def test_analyze_audio_timings(self):
        t0 = time.perf_counter()
        beat_chunks, D, timings, _ = jukebox.analyze_audio(self.audio, self.sr)
        elapsed = time.perf_counter() - t0
        print("Timings:", timings)
        self.assertIn('chroma', timings)
        self.assertIn('mfcc', timings)
        self.assertIn('beat', timings)
        self.assertLess(timings['chroma'], 1.0)
        self.assertLess(timings['mfcc'], 1.0)
        self.assertLess(timings['beat'], 1.0)
        self.assertLess(elapsed, 3.0)

if __name__ == "__main__":
    unittest.main()
