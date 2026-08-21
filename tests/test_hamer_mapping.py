import unittest

import numpy as np

from cv_experiment.estimators.hamer_mapping import (
    HAMER_OUTPUT_TO_PROJECT_21,
    JOINT_PROVENANCE,
    MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21,
    map_hamer_output,
    map_mano_16_plus_tips,
)


class HaMeRMappingTests(unittest.TestCase):
    def test_official_mano_mapping_has_all_21_unique_indices(self) -> None:
        self.assertEqual(len(MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21), 21)
        self.assertEqual(set(MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21), set(range(21)))

    def test_raw_mano_mapping_matches_semantic_order(self) -> None:
        raw = np.repeat(np.arange(21, dtype=np.float32)[:, None], 3, axis=1)
        mapped = map_mano_16_plus_tips(raw)
        np.testing.assert_array_equal(
            mapped[:, 0], np.asarray(MANO_16_PLUS_TIPS_TO_MEDIAPIPE_21)
        )

    def test_official_hamer_output_is_explicit_identity(self) -> None:
        points = np.arange(63, dtype=np.float32).reshape(21, 3)
        self.assertEqual(HAMER_OUTPUT_TO_PROJECT_21, tuple(range(21)))
        np.testing.assert_array_equal(map_hamer_output(points), points)

    def test_every_joint_has_explicit_native_or_vertex_provenance(self) -> None:
        self.assertEqual(len(JOINT_PROVENANCE), 21)
        tips = {4: 744, 8: 320, 12: 443, 16: 554, 20: 671}
        for record in JOINT_PROVENANCE:
            joint_id = record["joint_id"]
            self.assertFalse(record["derived"])
            if joint_id in tips:
                self.assertEqual(record["source_kind"], "mano_mesh_vertex")
                self.assertEqual(record["source_index"], tips[joint_id])
            else:
                self.assertEqual(record["source_kind"], "native_mano_joint")

    def test_invalid_shape_and_nan_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            map_hamer_output(np.zeros((16, 3), dtype=np.float32))
        points = np.zeros((21, 3), dtype=np.float32)
        points[4, 0] = np.nan
        with self.assertRaises(ValueError):
            map_hamer_output(points)


if __name__ == "__main__":
    unittest.main()
