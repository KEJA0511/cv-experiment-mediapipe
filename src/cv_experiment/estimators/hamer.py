"""Optional adapter for the official pretrained HaMeR implementation.

Heavy HaMeR/PyTorch imports are deliberately lazy, so the default MediaPipe
pipeline remains installable and usable without the separate HaMeR runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .base import EstimationResult, HandEstimate
from .hamer_mapping import map_hamer_output
from .mediapipe import MediaPipeEstimator


class HaMeRUnavailableError(RuntimeError):
    """Raised with setup guidance when the optional runtime cannot load."""


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


class HaMeREstimator:
    """MediaPipe 2D detection plus official HaMeR pretrained 3D inference."""

    backend_name = "hamer"

    def __init__(
        self,
        mediapipe_model_path: Path,
        *,
        hamer_root: Path,
        checkpoint_path: Path,
        device: str = "cuda",
        rescale_factor: float = 2.0,
    ) -> None:
        if not hamer_root.is_dir():
            raise HaMeRUnavailableError(f"HaMeR repository not found: {hamer_root}")
        if not checkpoint_path.is_file():
            raise HaMeRUnavailableError(
                f"HaMeR checkpoint not found: {checkpoint_path}. See docs/HAMER_SETUP.md"
            )
        mano_path = hamer_root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"
        if not mano_path.is_file():
            raise HaMeRUnavailableError(
                "Licensed MANO asset not found: "
                f"{mano_path}. Download MANO_RIGHT.pkl from the official MANO "
                "website after accepting its licence; do not use a mirror or dummy file."
            )

        root_string = str(hamer_root.resolve())
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        # Official HaMeR defaults pyrender to Linux EGL during module import.
        # Its renderer is not used here, but the eager import still needs a
        # valid Windows OpenGL loader.
        if sys.platform == "win32":
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
        try:
            import torch
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.configs import get_config
            from hamer.models import HAMER
            from hamer.utils import recursive_to
        except (ImportError, OSError) as error:
            raise HaMeRUnavailableError(
                "Official HaMeR dependencies could not be imported. "
                "Use the isolated Python 3.10 environment in docs/HAMER_SETUP.md. "
                f"Original error: {error}"
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise HaMeRUnavailableError(
                f"Requested device '{device}', but CUDA is unavailable. "
                "Pass --device cpu for a slow smoke test or configure CUDA."
            )

        original_working_directory = Path.cwd()
        try:
            # Official HaMeR resolves CACHE_DIR_HAMER as "./_DATA". Load from
            # its repository root, then restore the caller's working directory.
            os.chdir(hamer_root.resolve())
            # Licensed MANO pickles contain chumpy objects. chumpy 0.70 still
            # imports NumPy aliases removed in NumPy 1.24; restore only those
            # legacy names before unpickling, without changing numeric data.
            legacy_numpy_aliases = {
                "bool": bool,
                "int": int,
                "float": float,
                "complex": complex,
                "object": object,
                "unicode": str,
                "str": str,
            }
            for name, value in legacy_numpy_aliases.items():
                if name not in np.__dict__:
                    setattr(np, name, value)
            resolved_checkpoint = checkpoint_path.resolve()
            config_path = resolved_checkpoint.parent.parent / "model_config.yaml"
            model_cfg = get_config(str(config_path), update_cachedir=True)
            if (
                model_cfg.MODEL.BACKBONE.TYPE == "vit"
                and "BBOX_SHAPE" not in model_cfg.MODEL
            ):
                model_cfg.defrost()
                if model_cfg.MODEL.IMAGE_SIZE != 256:
                    raise ValueError(
                        f"Unexpected HaMeR image size: {model_cfg.MODEL.IMAGE_SIZE}"
                    )
                model_cfg.MODEL.BBOX_SHAPE = [192, 256]
                model_cfg.freeze()
            if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
                model_cfg.defrost()
                model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
                model_cfg.freeze()
            model = HAMER.load_from_checkpoint(
                str(resolved_checkpoint),
                strict=False,
                cfg=model_cfg,
                init_renderer=False,
            )
            model = model.to(device).eval()
        except Exception as error:
            raise HaMeRUnavailableError(
                "HaMeR model initialization failed. Confirm the checkpoint and "
                f"licensed MANO_RIGHT.pkl assets. Original error: {error}"
            ) from error
        finally:
            os.chdir(original_working_directory)

        self._torch = torch
        self._dataset_type = ViTDetDataset
        self._recursive_to = recursive_to
        self._model = model
        self._model_cfg = model_cfg
        self._device = device
        self._rescale_factor = rescale_factor
        self._checkpoint_path = checkpoint_path
        self._mediapipe = MediaPipeEstimator(mediapipe_model_path)

    @staticmethod
    def _boxes(hands: list[HandEstimate], width: int, height: int) -> np.ndarray:
        boxes = []
        for hand in hands:
            xy = np.clip(hand.image_joints[:, :2], 0.0, 1.0)
            minimum = xy.min(axis=0) * [width, height]
            maximum = xy.max(axis=0) * [width, height]
            boxes.append([minimum[0], minimum[1], maximum[0], maximum[1]])
        return np.asarray(boxes, dtype=np.float32)

    def estimate(self, rgb_image: np.ndarray) -> EstimationResult:
        start = perf_counter()
        mp_result = self._mediapipe.estimate(rgb_image)
        if not mp_result.hands:
            return EstimationResult(
                hands=[],
                estimator_metadata=self._metadata(mp_result.inference_time_ms, 0.0),
                inference_time_ms=(perf_counter() - start) * 1000.0,
            )

        height, width = rgb_image.shape[:2]
        boxes = self._boxes(mp_result.hands, width, height)
        is_right = np.asarray(
            [1.0 if hand.handedness == "Right" else 0.0 for hand in mp_result.hands],
            dtype=np.float32,
        )
        # The official demo passes cv2.imread output to ViTDetDataset (BGR).
        bgr_image = rgb_image[:, :, ::-1].copy()
        dataset = self._dataset_type(
            self._model_cfg,
            bgr_image,
            boxes,
            is_right,
            rescale_factor=self._rescale_factor,
        )
        loader = self._torch.utils.data.DataLoader(
            dataset, batch_size=len(mp_result.hands), shuffle=False, num_workers=0
        )

        hamer_start = perf_counter()
        batch = next(iter(loader))
        batch = self._recursive_to(batch, self._device)
        with self._torch.no_grad():
            output = self._model(batch)
        hamer_ms = (perf_counter() - hamer_start) * 1000.0

        predicted = output["pred_keypoints_3d"].detach().cpu().numpy()
        batch_right = batch["right"].detach().cpu().numpy()
        hands: list[HandEstimate] = []
        for index, mp_hand in enumerate(mp_result.hands):
            joints = map_hamer_output(predicted[index])
            # Official demo mirrors the model-space x coordinate for left hands.
            joints[:, 0] *= 2.0 * float(batch_right[index]) - 1.0
            details: dict[str, Any] = {}
            for key in ("pred_cam", "pred_cam_t", "pred_mano_params"):
                if key in output:
                    value = output[key]
                    if isinstance(value, dict):
                        details[key] = {
                            name: _json_value(component[index])
                            for name, component in value.items()
                        }
                    else:
                        details[key] = _json_value(value[index])
            if "pred_vertices" in output:
                details["mesh_vertex_count"] = int(output["pred_vertices"].shape[1])
            hands.append(
                HandEstimate(
                    image_joints=mp_hand.image_joints,
                    spatial_joints=joints,
                    handedness=mp_hand.handedness,
                    handedness_score=mp_hand.handedness_score,
                    spatial_metadata={
                        "source": "hamer",
                        "type": "hamer_mano_local_model_space",
                        "coordinate_frame": (
                            "right_mano_local_x_mirrored_for_left_hand"
                            if not bool(batch_right[index])
                            else "right_mano_local"
                        ),
                        "unit": "model_unit",
                        "unit_contract": "not_declared_metric_by_hamer_output_api",
                        "root_joint_id": 0,
                        "root_is_coordinate_origin": False,
                        "camera_translation_applied": False,
                        "scale_convention": "native_mano_shape_scale",
                        "canonical_scale_convention": (
                            "distance(wrist_0,middle_mcp_9)=1"
                        ),
                        "equivalent_to_mediapipe_world": False,
                        "joint_mapping": "official_hamer_21_to_mediapipe_hand_21",
                    },
                    backend_details=details,
                )
            )

        total_ms = (perf_counter() - start) * 1000.0
        return EstimationResult(
            hands=hands,
            estimator_metadata=self._metadata(mp_result.inference_time_ms, hamer_ms),
            inference_time_ms=total_ms,
        )

    def _metadata(self, mediapipe_ms: float, hamer_ms: float) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "pretrained": True,
            "checkpoint_path": self._checkpoint_path.as_posix(),
            "device": self._device,
            "image_landmarks_source": "mediapipe",
            "spatial_landmarks_source": "hamer",
            "stages_ms": {"mediapipe_2d": mediapipe_ms, "hamer_3d": hamer_ms},
        }

    def close(self) -> None:
        self._mediapipe.close()

    def __enter__(self) -> "HaMeREstimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
