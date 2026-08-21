import os
import unittest
from pathlib import Path

import numpy as np

from cv_experiment.hand_skeleton import records_to_array
from cv_experiment.pipeline import detect_image


class HaMeRIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("HAMER_ROOT") and os.environ.get("HAMER_CHECKPOINT"),
        "Set HAMER_ROOT and HAMER_CHECKPOINT after installing official HaMeR",
    )
    def test_official_pretrained_smoke(self) -> None:
        document = detect_image(
            Path("test_image.jpg"),
            Path("hand_landmarker.task"),
            backend="hamer",
            hamer_root=Path(os.environ["HAMER_ROOT"]),
            hamer_checkpoint=Path(os.environ["HAMER_CHECKPOINT"]),
            device=os.environ.get("HAMER_DEVICE", "cuda"),
        )
        self.assertEqual(document["estimator"]["backend"], "hamer")
        self.assertEqual(document["schema_version"], "2.0.0")
        self.assertEqual(document["topology"]["id"], "mediapipe_hand_21")
        self.assertGreater(document["estimator"]["inference_time_ms"], 0.0)
        self.assertEqual(document["estimator"]["image_landmarks_source"], "mediapipe")
        self.assertEqual(document["estimator"]["spatial_landmarks_source"], "hamer")

        hands = [hand for hand in document["frames"][0]["hands"] if hand["present"]]
        self.assertGreaterEqual(len(hands), 1)
        for hand in hands:
            skeleton = hand["skeleton"]
            image = records_to_array(skeleton["image_landmarks"])
            spatial = records_to_array(skeleton["world_landmarks"])
            canonical = records_to_array(skeleton["canonical_landmarks"])
            self.assertEqual(image.shape, (21, 3))
            self.assertEqual(spatial.shape, (21, 3))
            self.assertEqual(canonical.shape, (21, 3))
            self.assertTrue(np.isfinite(spatial).all())
            self.assertTrue(np.isfinite(canonical).all())
            np.testing.assert_allclose(canonical[0], 0.0, atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(canonical[9])), 1.0, places=5)
            self.assertIn(hand["validation"]["status"], {"accept", "review"})
            metadata = skeleton["world_landmarks_metadata"]
            self.assertEqual(metadata["source"], "hamer")
            self.assertFalse(metadata["equivalent_to_mediapipe_world"])
            self.assertFalse(metadata["camera_translation_applied"])


if __name__ == "__main__":
    unittest.main()
