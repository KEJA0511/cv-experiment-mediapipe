"""Concise runtime verification for MediaPipe and optional HaMeR assets."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    hamer_root = Path(os.environ.get("HAMER_ROOT", "D:/Code/third_party/hamer"))
    checkpoint = Path(
        os.environ.get(
            "HAMER_CHECKPOINT",
            str(hamer_root / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
        )
    )
    mano = hamer_root / "_DATA/data/mano/MANO_RIGHT.pkl"

    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    for package in ("mediapipe", "numpy", "opencv-python", "torch", "torchvision"):
        print(f"{package}: {_version(package)}")

    try:
        import torch

        available = torch.cuda.is_available()
        print(f"CUDA available: {available}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0) if available else 'none'}")
    except ImportError:
        print("CUDA available: False (torch not installed)")

    print(f"HaMeR root: {hamer_root} ({'present' if hamer_root.is_dir() else 'missing'})")
    print(f"HaMeR checkpoint: {checkpoint} ({'present' if checkpoint.is_file() else 'missing'})")
    print(f"MANO_RIGHT.pkl: {mano} ({'present' if mano.is_file() else 'missing'})")

    if hamer_root.is_dir():
        if sys.platform == "win32":
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
        root_string = str(hamer_root.resolve())
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            from hamer.models import load_hamer  # noqa: F401
            from hamer.datasets.vitdet_dataset import ViTDetDataset  # noqa: F401

            print("HaMeR imports: ok")
        except Exception as error:
            print(f"HaMeR imports: failed ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
