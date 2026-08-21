import tempfile
import unittest
from pathlib import Path

import numpy as np

from cv_experiment.estimators.base import EstimationResult, HandEstimate
from cv_experiment.hand_skeleton import records_to_array
from cv_experiment.render_skeleton import DEFAULT_POSE, render_document
from cv_experiment.skeleton import frame_from_estimation, make_document
from cv_experiment.training import document_features


def _document(backend: str) -> dict:
    spatial = DEFAULT_POSE.copy()
    image = spatial.copy() * 0.08
    image[:, 0] += 0.5
    image[:, 1] += 0.5
    result = EstimationResult(
        hands=[
            HandEstimate(
                image_joints=image,
                spatial_joints=spatial,
                handedness="Right",
                handedness_score=0.99,
                spatial_metadata={
                    "source": backend,
                    "unit": "meter" if backend == "mediapipe" else "model_unit",
                    "equivalent_to_mediapipe_world": backend == "mediapipe",
                },
            )
        ],
        estimator_metadata={
            "backend": backend,
            "image_landmarks_source": "mediapipe",
            "spatial_landmarks_source": backend,
        },
        inference_time_ms=12.5,
    )
    frame = frame_from_estimation(result, frame_index=0, timestamp_ms=0)
    metadata = dict(result.estimator_metadata)
    metadata["inference_time_ms"] = result.inference_time_ms
    return make_document([frame], source={"type": "test"}, estimator=metadata)


class BackendContractTests(unittest.TestCase):
    def test_both_backends_share_schema_normalization_and_features(self) -> None:
        for backend in ("mediapipe", "hamer"):
            with self.subTest(backend=backend):
                document = _document(backend)
                self.assertEqual(document["schema_version"], "2.0.0")
                self.assertEqual(document["topology"]["joint_count"], 21)
                self.assertEqual(document["estimator"]["backend"], backend)
                hands = {hand["slot"]: hand for hand in document["frames"][0]["hands"]}
                self.assertFalse(hands["Left"]["present"])
                self.assertTrue(hands["Left"]["renderer"]["uses_default_pose"])
                canonical = records_to_array(
                    hands["Right"]["skeleton"]["canonical_landmarks"]
                )
                np.testing.assert_allclose(canonical[0], 0.0, atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.norm(canonical[9])), 1.0, places=6)
                self.assertEqual(document_features(document).shape, (130,))

    def test_renderer_accepts_each_backend_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("mediapipe", "hamer"):
                path = Path(directory) / f"{backend}.png"
                render_document(_document(backend), path)
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
