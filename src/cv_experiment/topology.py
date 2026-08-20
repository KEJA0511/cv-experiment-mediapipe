"""Canonical MediaPipe 21-point hand graph topology."""

from __future__ import annotations

from typing import Any


TOPOLOGY_ID = "mediapipe_hand_21"

JOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# MediaPipe HAND_CONNECTIONS. In particular, the wrist connects to index MCP
# through (0, 5); (1, 5) appeared in an older Python implementation bug.
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def topology_definition() -> dict[str, Any]:
    return {
        "id": TOPOLOGY_ID,
        "joint_count": len(JOINT_NAMES),
        "bone_count": len(BONES),
        "joints": [
            {"id": joint_id, "name": name}
            for joint_id, name in enumerate(JOINT_NAMES)
        ],
        "bones": [
            {"id": bone_id, "start_joint_id": start, "end_joint_id": end}
            for bone_id, (start, end) in enumerate(BONES)
        ],
    }
