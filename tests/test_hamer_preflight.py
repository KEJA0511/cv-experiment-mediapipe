import tempfile
import unittest
from pathlib import Path

from cv_experiment.estimators.hamer import HaMeREstimator, HaMeRUnavailableError


class HaMeRPreflightTests(unittest.TestCase):
    def test_missing_licensed_mano_asset_fails_before_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "hamer.ckpt"
            checkpoint.touch()
            with self.assertRaisesRegex(HaMeRUnavailableError, "MANO_RIGHT.pkl"):
                HaMeREstimator(
                    root / "mediapipe.task",
                    hamer_root=root,
                    checkpoint_path=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
