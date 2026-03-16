Hands Off - Gesture-Controlled Infinite Jukebox
================================================

A song is analyzed into beats, clustered by harmonic and timbral similarity,
and played back as an infinite stream. Hand gestures captured via webcam
reshape that stream in real time.


HOW TO RUN
----------
    source venv/bin/activate
    python main.py media/[mediafile.mp3]

Optional flags:
    --hop N     Hop length for audio analysis (default: 512)
    --seed N    Random seed for reproducibility
    --k N       Number of beat clusters (default: 4)

Press Q or Ctrl+C to quit.


GESTURE CONTROLS
----------------

  Left hand
  ---------
  Fist               Loop the current beat continuously
  Pinch + rotate CW  Open the low-pass filter (brighter)
  Pinch + rotate CCW Close the low-pass filter (more muffled)

  Right hand
  ----------
  Fist               Eighth-note stutter (loops 1/8 of the current beat)
  Pinch + rotate CW  Increase tightness (more cluster jumping)
  Pinch + rotate CCW Decrease tightness (more sequential playback)
  1 finger (hold 2s) Select cluster 1
  2 fingers (hold 2s) Select cluster 2
  3 fingers (hold 2s) Select cluster 3
  4 fingers (hold 2s) Select cluster 4

Left fist takes priority over right fist when both are held.

Pinch knobs work like rotary encoders: the value adjusts relative to
rotation from wherever you grab. 60° of rotation covers the full range.


CLUSTER SELECTION
-----------------
The song's beats are grouped into k clusters (default 4) by harmonic and
timbral similarity. Hold 1–4 fingers on your right hand for 2 seconds to
activate that cluster. A progress bar on screen shows how long you've held.

Active clusters control where tightness jumps land and (at tightness > 0)
restrict sequential playback from crossing into inactive clusters.

Tightness:
  0%    Fully sequential - plays through the song normally, ignoring clusters
  100%  Always jumps to a random beat within the active clusters
  In between: blends sequential and cluster-jump behavior proportionally


DISPLAY
-------
Left panel  - webcam with hand skeleton, gesture labels, LPF %, Tight %,
              active cluster readout, and hold-progress bar
Right panel - PCA scatter plot of all beats colored by cluster; active
              cluster beats are highlighted, inactive are dimmed; cluster
              numbers labeled at centroids; current beat shown with a
              pulsing ring


DEPENDENCIES
------------
See requirements.txt. Install with:
    pip install -r requirements.txt

MediaPipe model file required at:
    models/hand_landmarker.task

Download from:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
