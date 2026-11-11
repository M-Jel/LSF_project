"""Extraction des points avec MediaPipe Tasks (HolisticLandmarker) au format ISLR.

Points d'attention (NON verifies, a mesurer) :
- ISLR a ete extrait avec l'ancienne API Holistic. Le visage Tasks renvoie 478
  points ; on garde les 468 premiers, en supposant que ce sont ceux de FaceMesh
  (les 10 derniers etant l'iris).
- La convention gauche/droite des mains peut differer entre les deux API. Le
  pretraitement « main la plus presente -> position droite » limite l'impact.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from lsf.features.landmarks import OFFSETS, ROWS_PER_FRAME

HOLISTIC_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
DEFAULT_MODEL = Path("models") / "holistic_landmarker.task"


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"telechargement de {HOLISTIC_URL} (environ 14 Mo)")
        urllib.request.urlretrieve(HOLISTIC_URL, path)
    return path


def result_to_frame(result) -> np.ndarray:
    """HolisticLandmarkerResult -> (543, 3), NaN pour les parties non detectees."""
    frame = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    parts = {
        "face": (result.face_landmarks, 468),
        "left_hand": (result.left_hand_landmarks, 21),
        "pose": (result.pose_landmarks, 33),
        "right_hand": (result.right_hand_landmarks, 21),
    }
    for name, (lms, n) in parts.items():
        if lms:
            # Selon la version, la liste peut etre imbriquee (une par personne).
            if isinstance(lms[0], list):
                lms = lms[0]
            arr = np.array([[p.x, p.y, p.z] for p in lms[:n]], dtype=np.float32)
            frame[OFFSETS[name] : OFFSETS[name] + len(arr)] = arr
    return frame


class HolisticExtractor:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(Path(model_path)))),
            running_mode=vision.RunningMode.VIDEO,
        )
        self._lm = vision.HolisticLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return result_to_frame(self._lm.detect_for_video(image, timestamp_ms))

    def close(self) -> None:
        self._lm.close()
