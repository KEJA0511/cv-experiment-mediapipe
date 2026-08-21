"""Build schema-v2 JSON documents from MediaPipe hand detections."""

from __future__ import annotations

from typing import Any

from .estimators.base import EstimationResult
from .hand_skeleton import HandSkeleton
from .topology import BONES as EDGES
from .topology import JOINT_NAMES as NODE_NAMES
from .topology import TOPOLOGY_ID, topology_definition
from .validation import validate_skeleton

SCHEMA_VERSION = "2.0.0"


def skeleton_definition() -> dict[str, Any]:
    """Backward-compatible alias for the canonical topology definition."""
    return topology_definition()


def _handedness(result: Any, hand_index: int) -> tuple[str | None, float | None]:
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return None, None
    category = result.handedness[hand_index][0]
    return category.category_name, float(category.score)


def _missing_hand(hand_id: int, slot: str) -> dict[str, Any]:
    return {
        "hand_id": hand_id,
        "slot": slot,
        "present": False,
        "tracking_state": "missing",
        "detection": {"handedness": None, "handedness_score": None},
        "binding": {"topology_id": TOPOLOGY_ID},
        "skeleton": None,
        "validation": {
            "valid": False,
            "status": "missing",
            "quality_score": None,
            "components": {},
            "issues": ["hand_not_detected"],
        },
        "renderer": {"uses_default_pose": True, "fallback_pose": "neutral_open_hand"},
    }


def frame_from_result(
    result: Any,
    *,
    frame_index: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    """Bind MediaPipe image/world landmarks into stable Left/Right slots."""
    detections = []
    world_results = getattr(result, "hand_world_landmarks", [])

    for hand_index, image_landmarks in enumerate(result.hand_landmarks):
        handedness, score = _handedness(result, hand_index)
        world_landmarks = (
            world_results[hand_index] if hand_index < len(world_results) else None
        )
        hand_skeleton = HandSkeleton.from_mediapipe(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
        )
        detections.append(
            {
                "handedness": handedness,
                "handedness_score": score,
                "skeleton": hand_skeleton,
                "validation": validate_skeleton(hand_skeleton),
            }
        )

    detections.sort(key=lambda hand: hand["handedness_score"] or 0.0, reverse=True)
    assigned: dict[str, dict[str, Any]] = {}
    for detection in detections:
        predicted = detection["handedness"]
        preferred = predicted if predicted in {"Left", "Right"} else None
        if preferred is not None and preferred not in assigned:
            assigned[preferred] = detection
            continue
        free_slot = next((slot for slot in ("Left", "Right") if slot not in assigned), None)
        if free_slot is not None:
            assigned[free_slot] = detection

    hands = []
    for hand_id, slot in enumerate(("Left", "Right")):
        detection = assigned.get(slot)
        if detection is None:
            hands.append(_missing_hand(hand_id, slot))
            continue
        hands.append(
            {
                "hand_id": hand_id,
                "slot": slot,
                "present": True,
                "tracking_state": "detected",
                "detection": {
                    "handedness": detection["handedness"],
                    "handedness_score": detection["handedness_score"],
                },
                "binding": {"topology_id": TOPOLOGY_ID},
                "skeleton": detection["skeleton"].to_json(),
                "validation": detection["validation"],
                "renderer": {"uses_default_pose": False, "fallback_pose": None},
            }
        )

    return {"frame_id": frame_index, "timestamp_ms": timestamp_ms, "hands": hands}


def frame_from_estimation(
    result: EstimationResult,
    *,
    frame_index: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    """Bind any estimator's 21-point output into stable Left/Right slots."""
    detections = []
    for estimate in result.hands:
        skeleton = HandSkeleton.from_arrays(
            estimate.image_joints,
            estimate.spatial_joints,
            spatial_metadata=estimate.spatial_metadata,
        )
        detections.append(
            {
                "handedness": estimate.handedness,
                "handedness_score": estimate.handedness_score,
                "skeleton": skeleton,
                "validation": validate_skeleton(skeleton),
                "backend_details": estimate.backend_details,
            }
        )

    detections.sort(key=lambda hand: hand["handedness_score"] or 0.0, reverse=True)
    assigned: dict[str, dict[str, Any]] = {}
    for detection in detections:
        preferred = detection["handedness"]
        if preferred in {"Left", "Right"} and preferred not in assigned:
            assigned[preferred] = detection
            continue
        free_slot = next((slot for slot in ("Left", "Right") if slot not in assigned), None)
        if free_slot is not None:
            assigned[free_slot] = detection

    hands = []
    for hand_id, slot in enumerate(("Left", "Right")):
        detection = assigned.get(slot)
        if detection is None:
            hands.append(_missing_hand(hand_id, slot))
            continue
        hands.append(
            {
                "hand_id": hand_id,
                "slot": slot,
                "present": True,
                "tracking_state": "detected",
                "detection": {
                    "handedness": detection["handedness"],
                    "handedness_score": detection["handedness_score"],
                },
                "binding": {"topology_id": TOPOLOGY_ID},
                "skeleton": detection["skeleton"].to_json(),
                "validation": detection["validation"],
                "estimator_output": detection["backend_details"],
                "renderer": {"uses_default_pose": False, "fallback_pose": None},
            }
        )

    return {"frame_id": frame_index, "timestamp_ms": timestamp_ms, "hands": hands}


def make_document(
    frames: list[dict[str, Any]],
    *,
    source: dict[str, Any] | None = None,
    estimator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = {
        "schema": {"name": "hand_skeleton", "version": SCHEMA_VERSION},
        "schema_version": SCHEMA_VERSION,
        "topology": topology_definition(),
        "coordinate_systems": {
            "image_landmarks": {
                "type": "mediapipe_normalized_image",
                "unit": "normalized",
                "origin": "image_top_left",
                "x_axis": "right",
                "y_axis": "down",
                "z_axis": "relative_depth_from_wrist",
            },
            "world_landmarks": {
                "type": "backend_spatial_3d",
                "unit": "backend_defined",
                "note": "See each hand's world_landmarks_metadata for provenance.",
            },
            "canonical_landmarks": {
                "type": "wrist_relative_palm_normalized",
                "unit": "palm_length",
                "origin_joint_id": 0,
                "scale_joint_pair": [0, 9],
            },
        },
        "frames": frames,
    }
    if source is not None:
        document["source"] = source
    if estimator is not None:
        document["estimator"] = estimator
    return document
