"""Backend-selectable inference entry points for the CV experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from .estimators import Hand3DEstimator, HaMeREstimator, MediaPipeEstimator
from .skeleton import SCHEMA_VERSION, frame_from_estimation, make_document

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_current_schema(path: Path) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return document.get("schema_version") == SCHEMA_VERSION


class HandDetector:
    """Backward-compatible reusable wrapper around a selected estimator."""

    def __init__(
        self,
        model_path: Path,
        *,
        backend: str = "mediapipe",
        hamer_root: Path | None = None,
        hamer_checkpoint: Path | None = None,
        device: str = "cuda",
    ) -> None:
        self._estimator = create_estimator(
            backend,
            model_path=model_path,
            hamer_root=hamer_root,
            hamer_checkpoint=hamer_checkpoint,
            device=device,
        )

    def close(self) -> None:
        self._estimator.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def detect(self, image_path: Path, *, source: dict[str, Any]) -> dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._estimator.estimate(rgb)
        frame = frame_from_estimation(result, frame_index=0, timestamp_ms=0)
        source_metadata = dict(source)
        source_metadata["width_px"] = int(image.shape[1])
        source_metadata["height_px"] = int(image.shape[0])
        estimator_metadata = dict(result.estimator_metadata)
        estimator_metadata["inference_time_ms"] = result.inference_time_ms
        return make_document(
            [frame], source=source_metadata, estimator=estimator_metadata
        )


def create_estimator(
    backend: str,
    *,
    model_path: Path,
    hamer_root: Path | None = None,
    hamer_checkpoint: Path | None = None,
    device: str = "cuda",
) -> Hand3DEstimator:
    """Create an estimator without importing optional HaMeR dependencies early."""
    if backend == "mediapipe":
        return MediaPipeEstimator(model_path)
    if backend == "hamer":
        if hamer_root is None or hamer_checkpoint is None:
            raise ValueError(
                "--backend hamer requires --hamer-root and --hamer-checkpoint"
            )
        return HaMeREstimator(
            model_path,
            hamer_root=hamer_root,
            checkpoint_path=hamer_checkpoint,
            device=device,
        )
    raise ValueError(f"Unknown backend: {backend}")


def detect_image(
    image_path: Path,
    model_path: Path,
    *,
    backend: str = "mediapipe",
    hamer_root: Path | None = None,
    hamer_checkpoint: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Detect hands in one image and return a JSON-serializable document."""
    with HandDetector(
        model_path,
        backend=backend,
        hamer_root=hamer_root,
        hamer_checkpoint=hamer_checkpoint,
        device=device,
    ) as detector:
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
    backend: str = "mediapipe",
    hamer_root: Path | None = None,
    hamer_checkpoint: Path | None = None,
    device: str = "cuda",
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

    stats = {
        "found": len(images),
        "converted": 0,
        "upgraded": 0,
        "skipped": 0,
        "no_hand": 0,
        "failed": 0,
    }
    with HandDetector(
        model_path,
        backend=backend,
        hamer_root=hamer_root,
        hamer_checkpoint=hamer_checkpoint,
        device=device,
    ) as detector:
        for index, image_path in enumerate(images, start=1):
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".json")

            if output_path.exists() and not overwrite:
                if _has_current_schema(output_path):
                    stats["skipped"] += 1
                    continue
                stats["upgraded"] += 1

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
