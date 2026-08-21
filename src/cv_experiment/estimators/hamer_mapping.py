"""Explicit mappings from official HaMeR/MANO joints to project topology."""

from __future__ import annotations

import numpy as np

from ..topology import JOINT_NAMES

# Official HaMeR MANO wrapper first concatenates MANO's 16 joints and five
# fingertip vertices, then applies this order to produce the standard 21 joints.
MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21 = (
    0, 13, 14, 15, 16,  # wrist + thumb
    1, 2, 3, 17,        # index
    4, 5, 6, 18,        # middle
    10, 11, 12, 19,     # ring
    7, 8, 9, 20,        # pinky
)

# pred_keypoints_3d from official HaMeR has already had the mapping above
# applied. The explicit identity prevents a silent topology assumption.
HAMER_OUTPUT_TO_PROJECT_21 = tuple(range(21))

# Provenance of every official HaMeR output joint. HaMeR's MANO wrapper first
# produces 16 native MANO joints, appends five surface vertices as fingertips,
# and applies MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21. No project joint is interpolated
# or otherwise derived. Vertex IDs come from smplx.vertex_ids['mano'].
_TIP_VERTEX_IDS = {
    16: 744,  # thumb
    17: 320,  # index
    18: 443,  # middle
    19: 554,  # ring
    20: 671,  # pinky
}

JOINT_PROVENANCE = tuple(
    {
        "joint_id": joint_id,
        "joint_name": JOINT_NAMES[joint_id],
        "hamer_output_index": joint_id,
        "mano_16_plus_tips_index": raw_index,
        "source_kind": (
            "mano_mesh_vertex" if raw_index in _TIP_VERTEX_IDS else "native_mano_joint"
        ),
        "source_index": _TIP_VERTEX_IDS.get(raw_index, raw_index),
        "derived": False,
    }
    for joint_id, raw_index in enumerate(MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21)
)


def _validated(points: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.shape != (len(JOINT_NAMES), 3):
        raise ValueError(f"{name} must have shape (21, 3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return value


def map_mano_16_plus_tips(points: np.ndarray) -> np.ndarray:
    """Map raw MANO 16 joints + 5 fingertip vertices to project order."""
    value = _validated(points, name="MANO 16+tips joints")
    return value[list(MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21)].copy()


def map_hamer_output(points: np.ndarray) -> np.ndarray:
    """Validate and bind official ``pred_keypoints_3d`` to project order."""
    value = _validated(points, name="HaMeR pred_keypoints_3d")
    return value[list(HAMER_OUTPUT_TO_PROJECT_21)].copy()
