"""Train a baseline classifier from generated hand-skeleton JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .hand_skeleton import records_to_array


def _hand_features(hand: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Return wrist-relative, scale-normalized features for one hand."""
    if not hand["present"]:
        return np.zeros(63, dtype=np.float32), None, 1.0

    if hand.get("skeleton") is not None:
        skeleton = hand["skeleton"]
        canonical = records_to_array(skeleton["canonical_landmarks"])
        image = records_to_array(skeleton["image_landmarks"])
        if canonical is None or image is None:
            return np.zeros(63, dtype=np.float32), None, 1.0
        image_scale = float(np.linalg.norm(image[9] - image[0]))
        return canonical.reshape(-1), image[0].copy(), max(image_scale, 1e-6)

    # Schema 1.x compatibility for previously generated datasets.
    if len(hand.get("nodes", [])) != 21:
        return np.zeros(63, dtype=np.float32), None, 1.0

    points = np.asarray(
        [
            [node["position"][axis] for axis in ("x", "y", "z")]
            for node in hand["nodes"]
        ],
        dtype=np.float32,
    )
    wrist = points[0].copy()
    relative = points - wrist
    palm_scale = float(np.linalg.norm(relative[9]))
    if palm_scale < 1e-6:
        palm_scale = 1.0
    return (relative / palm_scale).reshape(-1), wrist, palm_scale


def document_features(document: dict[str, Any]) -> np.ndarray:
    """Create a 130-value feature vector from a two-slot skeleton document."""
    hands = {hand["slot"]: hand for hand in document["frames"][0]["hands"]}
    left, left_wrist, left_scale = _hand_features(hands["Left"])
    right, right_wrist, right_scale = _hand_features(hands["Right"])
    masks = np.asarray(
        [float(hands["Left"]["present"]), float(hands["Right"]["present"])],
        dtype=np.float32,
    )

    wrist_delta = np.zeros(2, dtype=np.float32)
    if left_wrist is not None and right_wrist is not None:
        mean_scale = max((left_scale + right_scale) / 2.0, 1e-6)
        wrist_delta = (right_wrist[:2] - left_wrist[:2]) / mean_scale

    return np.concatenate((left, right, masks, wrist_delta))


def _group_id(label: str, source_path: str) -> str:
    """Keep flips and nearby sequential frames in the same split."""
    stem = Path(source_path).stem
    base_stem = re.sub(r"\s+flip$", "", stem, flags=re.IGNORECASE)
    match = re.search(r"(\d+)$", base_stem)
    if match:
        frame_number = int(match.group(1))
        return f"{label}/sequence_block_{frame_number // 50}"
    return f"{label}/{base_stem}"


def load_dataset(json_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    features = []
    labels = []
    groups = []
    stats = {"json_files": 0, "usable": 0, "zero_hand_skipped": 0}

    for path in sorted(json_dir.rglob("*.json")):
        stats["json_files"] += 1
        document = json.loads(path.read_text(encoding="utf-8"))
        source = document.get("source", {})
        label = source.get("label")
        source_path = source.get("path", path.as_posix())
        hands = document["frames"][0]["hands"]

        if not label:
            continue
        if not any(hand["present"] for hand in hands):
            stats["zero_hand_skipped"] += 1
            continue

        features.append(document_features(document))
        labels.append(label)
        groups.append(_group_id(label, source_path))
        stats["usable"] += 1

    if not features:
        raise ValueError(f"No usable skeleton JSON found in: {json_dir}")

    return (
        np.stack(features),
        np.asarray(labels),
        np.asarray(groups),
        stats,
    )


def stratified_group_holdout(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select whole groups per class while retaining every class in both sets."""
    rng = np.random.default_rng(random_state)
    validation_groups: set[str] = set()

    for label in np.unique(labels):
        label_groups = np.unique(groups[labels == label])
        if len(label_groups) < 2:
            raise ValueError(f"Class {label!r} needs at least two independent groups")
        rng.shuffle(label_groups)
        group_count = min(
            max(1, round(len(label_groups) * validation_fraction)),
            len(label_groups) - 1,
        )
        validation_groups.update(label_groups[:group_count].tolist())

    validation_mask = np.isin(groups, list(validation_groups))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def train_baseline(
    json_dir: Path,
    model_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    x, y, groups, data_stats = load_dataset(json_dir)
    train_indices, validation_indices = stratified_group_holdout(y, groups)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x[train_indices], y[train_indices])
    predictions = model.predict(x[validation_indices])
    classes = sorted(np.unique(y).tolist())

    metrics: dict[str, Any] = {
        "dataset": data_stats,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "classes": classes,
        "split_strategy": "per_class_20pct_group_holdout_sequence_blocks_of_50",
        "accuracy": float(accuracy_score(y[validation_indices], predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[validation_indices], predictions)
        ),
        "macro_f1": float(
            f1_score(y[validation_indices], predictions, average="macro")
        ),
        "classification_report": classification_report(
            y[validation_indices],
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y[validation_indices], predictions, labels=classes
        ).tolist(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": classes,
            "feature_schema": "two_hand_canonical_skeleton_v2",
            "feature_count": int(x.shape[1]),
        },
        model_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BSL skeleton baseline.")
    parser.add_argument("json_dir", type=Path, help="skeleton JSON dataset directory")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("output/models/bsl_baseline.joblib"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("output/metrics/bsl_baseline.json"),
    )
    args = parser.parse_args()

    metrics = train_baseline(args.json_dir, args.model_output, args.metrics_output)
    print(f"Classes: {', '.join(metrics['classes'])}")
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Validation samples: {metrics['validation_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Model written to: {args.model_output.resolve()}")
    print(f"Metrics written to: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()
