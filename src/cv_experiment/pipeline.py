"""MediaPipe inference entry points for the CV experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .skeleton import frame_from_result, make_document

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class HandDetector:
    """A reusable MediaPipe detector suitable for large image datasets."""

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

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def detect(self, image_path: Path, *, source: dict[str, Any]) -> dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        frame = frame_from_result(result, frame_index=0, timestamp_ms=0)
        return make_document([frame], source=source)


def detect_image(image_path: Path, model_path: Path) -> dict[str, Any]:
    """Detect hands in one image and return a JSON-serializable document."""
    with HandDetector(model_path) as detector:
        return detector.detect(
            image_path,
            source={"type": "image", "path": image_path.as_posix()},
        )


def write_json(document: dict[str, Any], output_path: Path) -> None:
    """Write a skeleton document as readable UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    *,
    limit: int | None = None,
    limit_per_class: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert a labelled image directory while preserving its structure."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Could not find dataset directory: {input_dir}")

    images = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit_per_class is not None:
        selected = []
        class_counts: dict[str, int] = {}
        for path in images:
            relative = path.relative_to(input_dir)
            label = relative.parts[0] if len(relative.parts) > 1 else "_root"
            if class_counts.get(label, 0) >= limit_per_class:
                continue
            selected.append(path)
            class_counts[label] = class_counts.get(label, 0) + 1
        images = selected
    elif limit is not None:
        images = images[:limit]

    stats = {"found": len(images), "converted": 0, "skipped": 0, "no_hand": 0, "failed": 0}
    with HandDetector(model_path) as detector:
        for index, image_path in enumerate(images, start=1):
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".json")

            if output_path.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            label = relative_path.parts[0] if len(relative_path.parts) > 1 else None
            try:
                document = detector.detect(
                    image_path,
                    source={
                        "type": "image",
                        "path": relative_path.as_posix(),
                        "label": label,
                    },
                )
                write_json(document, output_path)
                stats["converted"] += 1
                if not any(
                    hand["present"] for hand in document["frames"][0]["hands"]
                ):
                    stats["no_hand"] += 1
            except (OSError, ValueError):
                stats["failed"] += 1

            if index % 100 == 0 or index == len(images):
                print(f"Processed {index}/{len(images)} images")

    return stats
