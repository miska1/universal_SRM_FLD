from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from srm_steganalysis.extract import ExtractionConfig, extract_images
from srm_steganalysis.feature_io import (
    load_feature_dataset,
    parse_feature_directory,
    save_feature_dataset,
)
from srm_steganalysis.fld import FisherLDAEnsemble
from srm_steganalysis.logging_utils import configure_logging, elapsed_text
from srm_steganalysis.metrics import binary_metrics, select_balanced_threshold
from srm_steganalysis.model_io import load_model, save_model


class FeatureIOTests(unittest.TestCase):
    def test_parser_aligns_images_and_concatenates_sorted_submodels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "b.fea").write_text("5 6 second.pgm\n7 8 first.pgm\n", encoding="utf-8")
            (directory / "a.fea").write_text("1 first.pgm\n2 second.pgm\n", encoding="utf-8")
            result = parse_feature_directory(directory, ["first.pgm", "second.pgm"])
            np.testing.assert_array_equal(result.X, [[1, 7, 8], [2, 5, 6]])
            self.assertEqual(result.submodel_names, ("a", "b"))
            self.assertEqual(result.submodel_sizes, (1, 2))

    def test_feature_dataset_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            save_feature_dataset(
                path,
                np.asarray([[1, 2], [3, 4]], dtype=np.float32),
                np.asarray([0, 1], dtype=np.int8),
                ["cover/a.pgm", "stego/a.pgm"],
                ["one"],
                [2],
            )
            loaded = load_feature_dataset(path)
            np.testing.assert_array_equal(loaded["y"], [0, 1])
            self.assertEqual(np.asarray(loaded["files"]).tolist(), ["cover/a.pgm", "stego/a.pgm"])


class ExtractionConfigurationTests(unittest.TestCase):
    def test_worker_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            extract_images(
                [Path("unused.pgm")],
                Path("unused-extractor"),
                ExtractionConfig(workers=0),
            )


class FisherLDATests(unittest.TestCase):
    def test_model_learns_separable_data_and_serializes(self) -> None:
        rng = np.random.default_rng(8)
        X0 = rng.normal(-0.6, 1.0, size=(100, 24))
        X1 = rng.normal(0.6, 1.0, size=(100, 24))
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.r_[np.zeros(100, dtype=np.int8), np.ones(100, dtype=np.int8)]
        model = FisherLDAEnsemble(
            n_estimators=12,
            subspace_size=10,
            random_state=9,
        ).fit(X, y)
        scores = model.decision_function(X)
        model.set_threshold(select_balanced_threshold(y, scores))
        metrics = binary_metrics(y, scores, model.threshold_)
        self.assertGreater(metrics["accuracy"], 0.95)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pkl"
            save_model(path, model, {"schema_hash": "test"})
            restored, metadata = load_model(path)
            np.testing.assert_allclose(restored.decision_function(X), scores)
            self.assertEqual(metadata["schema_hash"], "test")

    def test_metric_definitions(self) -> None:
        y = np.asarray([0, 0, 1, 1], dtype=np.int8)
        scores = np.asarray([-1.0, 1.0, -1.0, 1.0])
        result = binary_metrics(y, scores, threshold=0.0)
        self.assertEqual(result["false_alarm_rate"], 0.5)
        self.assertEqual(result["missed_detection_rate"], 0.5)
        self.assertEqual(result["detection_probability"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1_score"], 0.5)


class LoggingTests(unittest.TestCase):
    def test_file_logging_and_duration_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            logger = configure_logging(path)
            logger.info("timed stage completed in %s", elapsed_text(1.25))
            for handler in logger.handlers:
                handler.flush()
            content = path.read_text(encoding="utf-8")
            self.assertIn("timed stage completed in 1.250 s", content)
            self.assertIn("INFO", content)


if __name__ == "__main__":
    unittest.main()
