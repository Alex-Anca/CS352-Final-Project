import unittest
import os
import pandas as pd
from PIL import Image

JESTER_PATH = os.path.join(os.path.dirname(__file__), '../jester_dataset/Train')
CSV_PATH = os.path.join(os.path.dirname(__file__), '../jester_dataset/Train.csv')


class TestJesterLoader(unittest.TestCase):
    def test_load_single_sample(self):
        df = pd.read_csv(CSV_PATH)
        # Use a specific, small video_id for testing (first row)
        row = df.iloc[0]
        video_id = str(row['video_id'])
        label = row['label']
        folder = os.path.join(JESTER_PATH, video_id)
        # Only check for the first 3 frames (should always exist)
        for i in range(1, 4):
            fname = f"{i:05d}.jpg"
            img_path = os.path.join(folder, fname)
            self.assertTrue(os.path.exists(img_path), f"Missing frame: {img_path}")
            img = Image.open(img_path)
            self.assertEqual(img.size, (176, 100))  # (width, height) from CSV shape
        print(f"Checked 3 frames for label '{label}' in video_id {video_id}")

if __name__ == "__main__":
    unittest.main()
