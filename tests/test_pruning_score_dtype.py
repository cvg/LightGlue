"""Native matcher precision controls without loading pretrained extractors."""

import importlib.util
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

# The matcher is self-contained; importing the package also loads extractors.
_SOURCE = Path(__file__).resolve().parents[1] / "lightglue" / "lightglue.py"
_SPEC = importlib.util.spec_from_file_location("native_lightglue_matcher", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LightGlue = _MODULE.LightGlue
DTYPES = (torch.float32, torch.float64, torch.float16, torch.bfloat16)


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def matcher(dtype, width_confidence=None, seed=7):
    torch.manual_seed(seed)
    overrides = (
        {} if width_confidence is None else {"width_confidence": width_confidence}
    )
    return (
        LightGlue(
            features=None,
            input_dim=32,
            descriptor_dim=32,
            n_layers=2,
            num_heads=4,
            flash=False,
            depth_confidence=-1,
            filter_threshold=0,
            **overrides,
        )
        .eval()
        .to(dtype=dtype)
    )


def data(dtype, counts=(4, 5), seed=107):
    torch.manual_seed(seed)
    return {
        f"image{i}": {
            "keypoints": (torch.rand(1, count, 2, dtype=torch.float32) * 24).to(dtype),
            "descriptors": F.normalize(
                torch.randn(1, count, 32, dtype=torch.float32), dim=-1
            ).to(dtype),
            "image_size": torch.tensor([[32.0, 24.0]], dtype=dtype),
        }
        for i, count in enumerate(counts)
    }


class TestPruningScoreDtype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_default_pruning_matches_disabled_pruning(self):
        for dtype in DTYPES:
            for global_dtype in (torch.float32, torch.float64):
                with self.subTest(dtype=dtype, global_dtype=global_dtype):
                    with default_dtype(global_dtype):
                        dense = matcher(dtype, -1)
                        pruned = matcher(dtype)
                        self.assertEqual(pruned.conf.width_confidence, 0.99)
                        pruned.load_state_dict(dense.state_dict())
                        features = data(dtype)
                        with torch.no_grad():
                            expected = dense(features)
                            actual = pruned(features)

                    for side in (0, 1):
                        key = f"matching_scores{side}"
                        self.assertEqual(expected[key].dtype, dtype)
                        self.assertEqual(actual[key].dtype, dtype)
                        torch.testing.assert_close(actual[key], expected[key])
                        torch.testing.assert_close(
                            actual[f"matches{side}"], expected[f"matches{side}"]
                        )
                        self.assertTrue(torch.all(actual[f"prune{side}"] == 2))
                    self.assertGreater(len(actual["matches"][0]), 0)

    def test_actual_pruning_retains_full_indexed_score_precision(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                model = matcher(dtype, 0.5, seed=2)
                features = data(dtype, counts=(8, 8), seed=2)
                with torch.no_grad():
                    actual = model(features)

                for side in (0, 1):
                    scores = actual[f"matching_scores{side}"]
                    removed = actual[f"prune{side}"] == 1
                    self.assertEqual(scores.dtype, dtype)
                    self.assertEqual(scores.shape, (1, 8))
                    self.assertTrue(torch.any(removed))
                    self.assertTrue(torch.any(~removed))
                    self.assertTrue(torch.all(scores[removed] == 0))
                    self.assertTrue(torch.all(actual[f"matches{side}"][removed] == -1))

    def test_empty_outputs_preserve_precision(self):
        for dtype in DTYPES:
            for counts in ((0, 5), (4, 0), (0, 0)):
                with self.subTest(dtype=dtype, counts=counts):
                    features = data(dtype, counts=counts)
                    with torch.no_grad():
                        actual = matcher(dtype)(features)
                    for side, count in enumerate(counts):
                        self.assertEqual(actual[f"matching_scores{side}"].dtype, dtype)
                        self.assertEqual(
                            actual[f"matching_scores{side}"].shape, (1, count)
                        )
                        self.assertTrue(
                            torch.all(actual[f"matching_scores{side}"] == 0)
                        )
                        self.assertTrue(torch.all(actual[f"matches{side}"] == -1))
                    self.assertEqual(actual["scores"].dtype, dtype)
                    self.assertEqual(actual["scores"].shape, (1, 0))

    def test_score_gradients_match_disabled_pruning(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                dense = matcher(dtype, -1)
                pruned = matcher(dtype)
                pruned.load_state_dict(dense.state_dict())
                features = data(dtype)
                for image in features.values():
                    image["descriptors"].requires_grad_()

                expected = dense(features)
                actual = pruned(features)
                expected_loss = sum(
                    expected[f"matching_scores{i}"].sum() for i in (0, 1)
                )
                actual_loss = sum(actual[f"matching_scores{i}"].sum() for i in (0, 1))
                expected_loss.backward()
                actual_loss.backward()

                self.assertTrue(
                    torch.any(dense.log_assignment[-1].final_proj.weight.grad != 0)
                )
                for expected_parameter, actual_parameter in zip(
                    dense.parameters(), pruned.parameters()
                ):
                    if expected_parameter.grad is None:
                        self.assertIsNone(actual_parameter.grad)
                    else:
                        self.assertIsNotNone(actual_parameter.grad)
                        torch.testing.assert_close(
                            actual_parameter.grad, expected_parameter.grad
                        )
                for image in features.values():
                    self.assertIsNone(image["descriptors"].grad)


if __name__ == "__main__":
    unittest.main()
