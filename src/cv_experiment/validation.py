"""Deterministic geometry validation for one hand skeleton."""

from __future__ import annotations

from typing import Any

import numpy as np

from .hand_skeleton import HandSkeleton


def validate_skeleton(skeleton: HandSkeleton) -> dict[str, Any]:
    issues: list[str] = []
    image_valid = (
        skeleton.image_joints.shape == (21, 3)
        and bool(np.all(np.isfinite(skeleton.image_joints)))
    )
    world_valid = (
        skeleton.world_joints is None
        or (
            skeleton.world_joints.shape == (21, 3)
            and bool(np.all(np.isfinite(skeleton.world_joints)))
        )
    )
    basic_score = 1.0 if image_valid and world_valid else 0.0
    if not image_valid:
        issues.append("invalid_image_landmarks")
    if not world_valid:
        issues.append("invalid_world_landmarks")

    lengths = skeleton.bone_lengths
    non_degenerate = np.isfinite(lengths) & (lengths >= 0.01) & (lengths <= 2.0)
    bone_score = float(non_degenerate.mean()) if len(lengths) else 0.0
    if skeleton.palm_scale < 1e-8:
        bone_score = 0.0
        issues.append("degenerate_palm_scale")
    elif bone_score < 1.0:
        issues.append("suspicious_bone_ratio")

    quality_score = 0.4 * basic_score + 0.6 * bone_score
    valid = basic_score == 1.0 and bone_score >= 0.75
    status = "accept" if valid and quality_score >= 0.8 else "review" if valid else "reject"
    return {
        "valid": valid,
        "status": status,
        "quality_score": quality_score,
        "components": {
            "basic": {"evaluated": True, "score": basic_score},
            "bone_integrity": {
                "evaluated": True,
                "score": bone_score,
                "normalized_lengths": [float(value) for value in lengths],
            },
            "joint_angle": {"evaluated": False, "score": None},
            "temporal": {"evaluated": False, "score": None},
            "reprojection": {"evaluated": False, "score": None},
        },
        "issues": issues,
    }
