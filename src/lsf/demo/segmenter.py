"""Palier 2 : decoupage « signe par signe » aux pauses (sans entrainement).

Une frame est « active » si au moins une main est visible ET bouge plus que
``motion_threshold`` (deplacement moyen des points de main, coordonnees image).
Un segment se ferme apres ``min_pause`` frames non actives ; il n'est emis que
s'il contient au moins ``min_sign`` frames actives.
"""

from __future__ import annotations

import numpy as np

from lsf.features.landmarks import OFFSETS

HAND_IDX = np.r_[OFFSETS["left_hand"] : OFFSETS["left_hand"] + 21, OFFSETS["right_hand"] : OFFSETS["right_hand"] + 21]


class PauseSegmenter:
    def __init__(self, min_pause: int = 8, min_sign: int = 6, motion_threshold: float = 0.004, max_frames: int = 150):
        self.min_pause, self.min_sign = min_pause, min_sign
        self.motion_threshold, self.max_frames = motion_threshold, max_frames
        self._buf: list[np.ndarray] = []
        self._active = 0
        self._idle = 0
        self._prev = None

    def _is_active(self, frame: np.ndarray) -> bool:
        hands = frame[HAND_IDX, :2]
        if np.isnan(hands).all():
            self._prev = None
            return False
        active = False
        if self._prev is not None:
            d = np.linalg.norm(hands - self._prev, axis=-1)
            d = d[~np.isnan(d)]
            # Une main qui apparait ou disparait compte comme un mouvement.
            appeared = np.isnan(self._prev[:, 0]) != np.isnan(hands[:, 0])
            active = bool(appeared.any()) or (d.size > 0 and float(d.mean()) > self.motion_threshold)
        self._prev = hands
        return active

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """Ajoute une frame (543, 3). Retourne un segment (T, 543, 3) quand un signe se termine."""
        if self._is_active(frame):
            self._buf.append(frame)
            self._active += 1
            self._idle = 0
        elif self._buf:
            self._buf.append(frame)
            self._idle += 1
        if self._buf and (self._idle >= self.min_pause or len(self._buf) >= self.max_frames):
            return self._flush()
        return None

    def _flush(self) -> np.ndarray | None:
        seg = self._buf[: len(self._buf) - self._idle] if self._idle else self._buf
        ok = self._active >= self.min_sign
        self._buf, self._active, self._idle = [], 0, 0
        return np.stack(seg) if ok and seg else None

    def flush(self) -> np.ndarray | None:
        return self._flush() if self._buf else None
