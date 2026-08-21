from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import convert_dataset, detect_image, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert hand estimates to shared 21-point skeleton JSON."
    )
    parser.add_argument("input", type=Path, help="input image or dataset directory")
    parser.add_argument(
        "--backend",
        choices=("mediapipe", "hamer"),
        default="mediapipe",
        help="3D estimator backend (default: mediapipe)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("hand_landmarker.task"),
        help="MediaPipe .task model path (default: hand_landmarker.task)",
    )
    parser.add_argument(
        "--hamer-root",
        type=Path,
        help="official HaMeR repository root (required for HaMeR)",
    )
    parser.add_argument(
        "--hamer-checkpoint",
        type=Path,
        help="official pretrained HaMeR checkpoint (required for HaMeR)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="HaMeR torch device, e.g. cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON file, or output directory in dataset mode",
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit",
        type=int,
        help="process only the first N dataset images (useful for testing)",
    )
    limit_group.add_argument(
        "--limit-per-class",
        type=int,
        help="process the first N images from every class directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace JSON files that already exist",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        output_dir = args.output or Path("data/interim/landmarks_json")
        stats = convert_dataset(
            args.input,
            output_dir,
            args.model,
            limit=args.limit,
            limit_per_class=args.limit_per_class,
            overwrite=args.overwrite,
            backend=args.backend,
            hamer_root=args.hamer_root,
            hamer_checkpoint=args.hamer_checkpoint,
            device=args.device,
        )
        print("Dataset conversion complete:")
        for name, value in stats.items():
            print(f"  {name}: {value}")
        print(f"JSON directory: {output_dir.resolve()}")
        return

    output_path = args.output or Path("output/hand_skeleton.json")
    document = detect_image(
        args.input,
        args.model,
        backend=args.backend,
        hamer_root=args.hamer_root,
        hamer_checkpoint=args.hamer_checkpoint,
        device=args.device,
    )
    write_json(document, output_path)

    hand_count = sum(
        hand["present"] for hand in document["frames"][0]["hands"]
    )
    print(f"Detected hands: {hand_count}")
    print(f"JSON written to: {output_path.resolve()}")
