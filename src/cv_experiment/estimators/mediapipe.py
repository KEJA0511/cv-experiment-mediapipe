"""MediaPipe implementation of the common estimator contract."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base import EstimationResult, HandEstimate


def _points(landmarks: Iterable[Any] | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    return np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


class MediaPipeEstimator:
    backend_name = "mediapipe"

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find model: {model_path}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self.model_path = model_path

    def estimate(self, rgb_image: np.ndarray) -> EstimationResult:
        start = perf_counter()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._detector.detect(mp_image)
        hands: list[HandEstimate] = []
        world_results = getattr(result, "hand_world_landmarks", [])
        handedness_results = getattr(result, "handedness", [])
        for index, image_landmarks in enumerate(result.hand_landmarks):
            category = (
                handedness_results[index][0]
                if index < len(handedness_results) and handedness_results[index]
                else None
            )
            spatial = _points(
                world_results[index] if index < len(world_results) else None
            )
            hands.append(
                HandEstimate(
                    image_joints=_points(image_landmarks),  # type: ignore[arg-type]
                    spatial_joints=spatial,
                    handedness=category.category_name if category else None,
                    handedness_score=float(category.score) if category else None,
                    spatial_metadata={
                        "source": "mediapipe",
                        "type": "mediapipe_world",
                        "unit": "meter",
                        "equivalent_to_mediapipe_world": True,
                    },
                )
            )
        elapsed = (perf_counter() - start) * 1000.0
        return EstimationResult(
            hands=hands,
            estimator_metadata={
                "backend": self.backend_name,
                "pretrained": True,
                "model_path": self.model_path.as_posix(),
                "image_landmarks_source": "mediapipe",
                "spatial_landmarks_source": "mediapipe",
            },
            inference_time_ms=elapsed,
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MediaPipeEstimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
