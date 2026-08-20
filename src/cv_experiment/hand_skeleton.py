"""In-memory representation of one MediaPipe hand skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .topology import BONES, JOINT_NAMES, TOPOLOGY_ID


def _points(landmarks: Iterable[Any] | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    return np.asarray(
        [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
    )


def _records(points: np.ndarray | None) -> list[dict[str, Any]] | None:
    if points is None:
        return None
    return [
        {
            "joint_id": joint_id,
            "position": {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            },
        }
        for joint_id, point in enumerate(points)
    ]


@dataclass(frozen=True)
class HandSkeleton:
    """A 21-joint graph with image, world and canonical coordinates."""

    image_joints: np.ndarray
    world_joints: np.ndarray | None

    @classmethod
    def from_mediapipe(
        cls,
        image_landmarks: Iterable[Any],
        world_landmarks: Iterable[Any] | None,
    ) -> "HandSkeleton":
        image = _points(image_landmarks)
        world = _points(world_landmarks)
        if image is None:
            raise ValueError("Image landmarks are required")
        return cls(image_joints=image, world_joints=world)

    @property
    def canonical_source(self) -> str:
        return "world_landmarks" if self.world_joints is not None else "image_landmarks"

    @property
    def canonical_source_joints(self) -> np.ndarray:
        return self.world_joints if self.world_joints is not None else self.image_joints

    @property
    def palm_scale(self) -> float:
        points = self.canonical_source_joints
        if points.shape != (21, 3):
            return 0.0
        return float(np.linalg.norm(points[9] - points[0]))

    @property
    def canonical_joints(self) -> np.ndarray:
        points = self.canonical_source_joints
        scale = self.palm_scale
        if points.shape != (21, 3) or scale < 1e-8:
            return np.zeros((21, 3), dtype=np.float32)
        return (points - points[0]) / scale

    @property
    def bone_lengths(self) -> np.ndarray:
        points = self.canonical_joints
        return np.asarray(
            [np.linalg.norm(points[start] - points[end]) for start, end in BONES],
            dtype=np.float32,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "topology_id": TOPOLOGY_ID,
            "image_landmarks": _records(self.image_joints),
            "world_landmarks": _records(self.world_joints),
            "canonical_landmarks": _records(self.canonical_joints),
            "normalization": {
                "source": self.canonical_source,
                "origin_joint_id": 0,
                "scale_joint_pair": [0, 9],
                "scale_value": self.palm_scale,
                "operation": "(joint - wrist) / distance(wrist, middle_mcp)",
            },
        }


def records_to_array(records: list[dict[str, Any]] | None) -> np.ndarray | None:
    """Read schema-v2 records into topology-indexed NumPy coordinates."""
    if records is None:
        return None
    points = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    seen = set()
    for record in records:
        joint_id = int(record["joint_id"])
        if joint_id < 0 or joint_id >= len(JOINT_NAMES) or joint_id in seen:
            raise ValueError(f"Invalid or duplicate joint_id: {joint_id}")
        position = record["position"]
        points[joint_id] = [position["x"], position["y"], position["z"]]
        seen.add(joint_id)
    if len(seen) != len(JOINT_NAMES):
        raise ValueError(f"Expected 21 joints, received {len(seen)}")
    return points
