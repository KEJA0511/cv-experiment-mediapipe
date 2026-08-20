"""Convert MediaPipe hand detections into the project's JSON schema."""

from __future__ import annotations

from typing import Any


NODE_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# MediaPipe HAND_CONNECTIONS. The palm is represented by a loop rather than
# connecting every finger base directly to the wrist.
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def skeleton_definition() -> dict[str, Any]:
    """Return the fixed node names and edge topology for a hand."""
    return {
        "node_names": list(NODE_NAMES),
        "edges": [list(edge) for edge in EDGES],
    }


def _handedness(result: Any, hand_index: int) -> tuple[str | None, float | None]:
    """Read handedness defensively because MediaPipe may omit it."""
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return None, None

    category = result.handedness[hand_index][0]
    return category.category_name, float(category.score)


def frame_from_result(
    result: Any,
    *,
    frame_index: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    """Convert one result into stable Left and Right renderer slots."""
    detections = []

    for hand_index, landmarks in enumerate(result.hand_landmarks):
        handedness, handedness_score = _handedness(result, hand_index)
        nodes = [
            {
                "id": node_id,
                "name": NODE_NAMES[node_id],
                "position": {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                },
            }
            for node_id, landmark in enumerate(landmarks)
        ]

        detections.append(
            {
                "handedness": handedness,
                "handedness_score": handedness_score,
                "nodes": nodes,
            }
        )

    # Assign the most confident detections first. If MediaPipe gives both hands
    # the same handedness, the second detection falls back to the free slot.
    detections.sort(
        key=lambda hand: hand["handedness_score"] or 0.0,
        reverse=True,
    )
    assigned: dict[str, dict[str, Any]] = {}
    for detection in detections:
        predicted = detection["handedness"]
        preferred_slot = predicted if predicted in {"Left", "Right"} else None
        if preferred_slot is not None and preferred_slot not in assigned:
            assigned[preferred_slot] = detection
            continue

        free_slot = next(
            (slot for slot in ("Left", "Right") if slot not in assigned),
            None,
        )
        if free_slot is not None:
            assigned[free_slot] = detection

    hands = []
    for hand_id, slot in enumerate(("Left", "Right")):
        detection = assigned.get(slot)
        if detection is None:
            hands.append(
                {
                    "hand_id": hand_id,
                    "slot": slot,
                    "present": False,
                    "handedness": None,
                    "handedness_score": None,
                    "uses_default_pose": True,
                    "nodes": [],
                }
            )
        else:
            hands.append(
                {
                    "hand_id": hand_id,
                    "slot": slot,
                    "present": True,
                    "handedness": detection["handedness"],
                    "handedness_score": detection["handedness_score"],
                    "uses_default_pose": False,
                    "nodes": detection["nodes"],
                }
            )

    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "hands": hands,
    }


def make_document(
    frames: list[dict[str, Any]],
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap frames with schema and coordinate-system metadata."""
    document = {
        "schema_version": "1.1",
        "coordinate_system": {
            "type": "mediapipe_normalized",
            "x": "right",
            "y": "down",
            "z": "relative_depth",
            "origin": "image_top_left",
        },
        "skeleton_definition": skeleton_definition(),
        "frames": frames,
    }
    if source is not None:
        document["source"] = source
    return document
