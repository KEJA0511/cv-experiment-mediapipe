"""Render the JSON hand graph as a bound 3D skeleton preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .skeleton import EDGES


DEFAULT_POSE = np.asarray(
    [
        [0.00, -1.00, 0.00],
        [-0.22, -0.78, 0.02], [-0.45, -0.52, 0.04], [-0.68, -0.24, 0.03], [-0.90, 0.02, 0.00],
        [-0.38, -0.35, 0.00], [-0.41, 0.12, 0.00], [-0.43, 0.53, 0.00], [-0.44, 0.88, 0.00],
        [0.00, -0.30, 0.00], [0.00, 0.22, 0.00], [0.00, 0.67, 0.00], [0.00, 1.04, 0.00],
        [0.34, -0.36, 0.00], [0.37, 0.12, 0.00], [0.39, 0.52, 0.00], [0.40, 0.84, 0.00],
        [0.63, -0.50, 0.00], [0.68, -0.08, 0.00], [0.71, 0.25, 0.00], [0.73, 0.53, 0.00],
    ],
    dtype=np.float32,
)


def _detected_points(hand: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [
                node["position"]["x"],
                -node["position"]["y"],
                -node["position"]["z"],
            ]
            for node in hand["nodes"]
        ],
        dtype=np.float32,
    )


def bound_poses(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind frame nodes to edges and supply neutral poses for missing slots."""
    hands = document["frames"][0]["hands"]
    present = [hand for hand in hands if hand["present"]]
    detected = {hand["slot"]: _detected_points(hand) for hand in present}

    if detected:
        wrists = np.stack([points[0] for points in detected.values()])
        origin = wrists.mean(axis=0)
        palm_scales = [
            float(np.linalg.norm(points[9] - points[0]))
            for points in detected.values()
        ]
        scale = max(float(np.mean(palm_scales)), 1e-6)
        detected = {
            slot: (points - origin) / scale for slot, points in detected.items()
        }

    poses = []
    for hand in hands:
        slot = hand["slot"]
        if hand["present"]:
            points = detected[slot]
        else:
            # A neutral renderer pose is visibly separated from a detected hand.
            direction = -1.0 if slot == "Left" else 1.0
            points = DEFAULT_POSE.copy()
            if slot == "Right":
                points[:, 0] *= -1
            points[:, 0] += direction * 2.1

        poses.append(
            {
                "slot": slot,
                "present": hand["present"],
                "points": points,
            }
        )
    return poses


def render_document(document: dict[str, Any], output_path: Path) -> None:
    poses = bound_poses(document)
    all_points = np.concatenate([pose["points"] for pose in poses], axis=0)
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    radius = max(float(np.ptp(all_points, axis=0).max()) / 2, 1.0) * 1.12

    source = document.get("source", {})
    source_name = Path(source.get("path", "skeleton JSON")).name
    views = ((18, -68, "Perspective"), (0, -90, "Front"), (5, 0, "Side / depth"))
    fig = plt.figure(figsize=(13, 4.4))

    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        for pose in poses:
            points = pose["points"]
            color = "#2563eb" if pose["slot"] == "Left" else "#dc2626"
            alpha = 1.0 if pose["present"] else 0.38
            line_style = "-" if pose["present"] else "--"
            for start, end in EDGES:
                segment = points[[start, end]]
                axis.plot(
                    segment[:, 0], segment[:, 1], segment[:, 2],
                    color=color, alpha=alpha, linestyle=line_style, linewidth=2,
                )
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                color=color, alpha=alpha, s=18,
                label=f"{pose['slot']} ({'detected' if pose['present'] else 'default'})",
            )

        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Depth")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        if index == 1:
            axis.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"JSON → bound 3D hand skeleton\n{source_name}", y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render skeleton JSON in 3D.")
    parser.add_argument("json", type=Path, help="input skeleton JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/3d_skeleton.png"),
        help="output PNG path",
    )
    args = parser.parse_args()

    document = json.loads(args.json.read_text(encoding="utf-8"))
    render_document(document, args.output)
    print(f"3D skeleton written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
