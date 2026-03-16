import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_cluster_data(features, k=8):
    """Cluster beats and project to 2D.

    Parameters
    ----------
    features : np.ndarray, shape (12, N)
        Beat-synced chroma features.
    k : int
        Number of clusters.

    Returns
    -------
    cluster_labels : np.ndarray (N,) int
    pca_coords     : np.ndarray (N, 2) float
    k_colors       : np.ndarray (k, 3) uint8  BGR
    """
    X = features.T  # (N, 12)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ac = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = ac.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    # evenly-spaced HSV hues -> BGR
    hsv = np.zeros((k, 1, 3), dtype=np.uint8)
    for i in range(k):
        hsv[i, 0, 0] = int(180 * i / k)
        hsv[i, 0, 1] = 220
        hsv[i, 0, 2] = 230
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(k, 3)

    return labels, coords, bgr


class ClusterRenderer:
    def __init__(self, pca_coords, cluster_labels, k_colors,
                 beat_chunk_lengths, canvas_size=(300, 300), n_candidates=5):
        self._coords = pca_coords        # (N, 2)
        self._labels = cluster_labels    # (N,)
        self._k_colors = k_colors        # (k, 3)
        self._chunk_lens = beat_chunk_lengths
        self._canvas_size = canvas_size
        self._n_candidates = n_candidates

        H, W = canvas_size
        margin = 10
        mn = pca_coords.min(axis=0)
        mx = pca_coords.max(axis=0)
        rng = mx - mn
        rng[rng == 0] = 1.0

        px = (pca_coords - mn) / rng  # (N, 2) in [0, 1]
        px[:, 0] = margin + px[:, 0] * (W - 2 * margin)
        px[:, 1] = margin + px[:, 1] * (H - 2 * margin)
        self._px = px.astype(np.int32)  # (N, 2) pixel coords

        # Pre-render static background: dim dots for all beats
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        for i, (x, y) in enumerate(self._px):
            color = tuple(int(c * 0.25) for c in k_colors[cluster_labels[i]])
            cv2.circle(bg, (x, y), 2, color, -1)
        self._static_bg = bg

        # Cluster centroids in pixel space + k count
        self._k = len(k_colors)
        self._centroids = {}
        for ci in range(self._k):
            mask = cluster_labels == ci
            if mask.any():
                self._centroids[ci] = self._px[mask].mean(axis=0).astype(int)

    def render(self, state) -> np.ndarray:
        """Return the rendered 300×300 canvas (no resizing or placement logic)."""
        canvas = self._static_bg.copy()

        # Overdraw active cluster beats at full brightness
        active = state.active_clusters
        for i, (x, y) in enumerate(self._px):
            if self._labels[i] in active:
                color = tuple(int(c) for c in self._k_colors[self._labels[i]])
                cv2.circle(canvas, (x, y), 3, color, -1)

        # Draw cluster number labels; highlight active centroids
        for ci, centroid in self._centroids.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            color = tuple(int(c) for c in self._k_colors[ci])
            if ci in active:
                cv2.circle(canvas, (cx, cy), 14, color, 2)
                cv2.putText(canvas, str(ci + 1), (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                cv2.putText(canvas, str(ci + 1), (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        current_beat = state.current_beat
        chunk_len = self._chunk_lens[current_beat] if current_beat < len(self._chunk_lens) else 1
        frac = min(state.position_in_beat / max(chunk_len, 1), 1.0)

        # Draw line from previous beat to current beat
        prev_beat = state.prev_beat
        if prev_beat is not None and 0 <= prev_beat < len(self._px) and 0 <= current_beat < len(self._px):
            px0 = tuple(self._px[prev_beat])
            px1 = tuple(self._px[current_beat])
            cv2.line(canvas, px0, px1, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw current beat with pulsing ring
        if 0 <= current_beat < len(self._px):
            x, y = self._px[current_beat]
            color = tuple(int(c) for c in self._k_colors[self._labels[current_beat]])
            cv2.circle(canvas, (x, y), 5, color, -1)
            pulse_r = int(7 + 4 * frac)
            cv2.circle(canvas, (x, y), pulse_r, (255, 255, 255), 1)

        return canvas
