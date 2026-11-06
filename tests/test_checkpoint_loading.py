"""Real-Torch tests using synthetic, randomly initialized matcher checkpoints.

Default mode imports the public LightGlue package. To test the complete matcher
module without importing optional extractor dependencies, set
LIGHTGLUE_TEST_MATCHER_PATH to the exact lightglue/lightglue.py under test. That
explicit mode does not validate public-package integration. Neither mode loads
official pretrained weights or establishes matching accuracy.

The small descriptor dimension reduces CPU/memory cost. Only checkpoint download
transport is replaced; model initialization, state loading, and forward are real.
Local-checkpoint tests also use real torch.save/torch.load, with a byte-identical
copy of the full matcher module in a temporary directory.
"""

import importlib
import importlib.util
import math
import os
import re
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

MODEL_CONF = {
    "input_dim": 128,
    "descriptor_dim": 32,
    "num_heads": 4,
    "flash": False,
}


def _load_complete_matcher(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_checkpoint_test_matcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the complete matcher module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clone_state(state):
    return {name: value.clone() for name, value in state.items()}


def _expected_thresholds(n_layers):
    return torch.tensor(
        [0.8 + 0.1 * math.exp(-4.0 * i / n_layers) for i in range(n_layers)],
        dtype=torch.float32,
    )


def _assert_state_unchanged(state, before):
    assert list(state) == list(before)
    for name in before:
        torch.testing.assert_close(state[name], before[name], rtol=0, atol=0)


def _assert_parameters_loaded(model, expected):
    parameters = dict(model.named_parameters())
    assert parameters
    for name, parameter in parameters.items():
        assert name in expected, name
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


def _download_fixture(monkeypatch, state):
    transport = Mock(return_value=state)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", transport)
    return transport


@pytest.fixture(scope="module")
def matcher_module():
    source = os.environ.get("LIGHTGLUE_TEST_MATCHER_PATH")
    if source:
        return _load_complete_matcher(source)
    return importlib.import_module("lightglue.lightglue")


@pytest.fixture(autouse=True)
def forbid_checkpoint_downloads(monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError(
            "These synthetic-checkpoint tests must not download weights"
        )

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", unexpected_download)
    monkeypatch.setattr(torch.hub, "download_url_to_file", unexpected_download)


@pytest.fixture(scope="module")
def checkpoint9(matcher_module):
    # No feature preset or weight name: this constructor never downloads weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(183)
        donor = matcher_module.LightGlue(features=None, n_layers=9, **MODEL_CONF)
    state = _clone_state(donor.state_dict())
    # A recognizable learned parameter prevents a no-op load from passing.
    state["input_proj.bias"] = torch.linspace(-0.3, 0.3, MODEL_CONF["descriptor_dim"])
    return state


@pytest.fixture
def local_matcher_module(matcher_module, tmp_path):
    source_bytes = Path(matcher_module.__file__).read_bytes()
    local_source = tmp_path / "lightglue.py"
    local_source.write_bytes(source_bytes)
    assert local_source.read_bytes() == source_bytes
    (tmp_path / "weights").mkdir()
    return _load_complete_matcher(local_source)


@pytest.mark.parametrize("n_layers", [1, 5, 9])
def test_feature_checkpoint_depth(matcher_module, checkpoint9, monkeypatch, n_layers):
    state = _clone_state(checkpoint9)
    before = _clone_state(state)
    transport = _download_fixture(monkeypatch, state)

    model = matcher_module.LightGlue(
        features="raco-aliked", n_layers=n_layers, **MODEL_CONF
    )

    transport.assert_called_once_with(
        model.url.format(model.version, "raco_aliked_lightglue"),
        file_name=f"raco_aliked_lightglue_{model.version.replace('.', '-')}.pth",
    )
    assert len(model.transformers) == n_layers
    assert len(model.log_assignment) == n_layers
    assert len(model.token_confidence) == n_layers - 1
    _assert_parameters_loaded(model, checkpoint9)
    torch.testing.assert_close(
        model.confidence_thresholds, _expected_thresholds(n_layers)
    )
    _assert_state_unchanged(state, before)
    if n_layers == 5:
        assert not torch.allclose(
            model.confidence_thresholds, checkpoint9["confidence_thresholds"][:5]
        )


@pytest.mark.parametrize("n_layers", [1, 5, 9])
def test_local_checkpoint_depth(local_matcher_module, checkpoint9, n_layers):
    checkpoint_path = (
        Path(local_matcher_module.__file__).parent / "weights/synthetic.pth"
    )
    torch.save(checkpoint9, checkpoint_path)

    model = local_matcher_module.LightGlue(
        features=None, weights="synthetic", n_layers=n_layers, **MODEL_CONF
    )

    _assert_parameters_loaded(model, checkpoint9)
    torch.testing.assert_close(
        model.confidence_thresholds, _expected_thresholds(n_layers)
    )


@pytest.mark.parametrize("n_layers", [1, 5, 9])
def test_checkpoint_without_thresholds(
    matcher_module, checkpoint9, monkeypatch, n_layers
):
    state = _clone_state(checkpoint9)
    del state["confidence_thresholds"]
    before = _clone_state(state)
    _download_fixture(monkeypatch, state)

    model = matcher_module.LightGlue(
        features="raco-aliked", n_layers=n_layers, **MODEL_CONF
    )

    _assert_parameters_loaded(model, checkpoint9)
    torch.testing.assert_close(
        model.confidence_thresholds, _expected_thresholds(n_layers)
    )
    _assert_state_unchanged(state, before)


def test_same_shape_checkpoint_thresholds_are_preserved(
    matcher_module, checkpoint9, monkeypatch
):
    state = _clone_state(checkpoint9)
    state["confidence_thresholds"] = torch.linspace(0.71, 0.79, 9)
    before = _clone_state(state)
    _download_fixture(monkeypatch, state)

    model = matcher_module.LightGlue(features="raco-aliked", n_layers=9, **MODEL_CONF)

    _assert_parameters_loaded(model, checkpoint9)
    torch.testing.assert_close(
        model.confidence_thresholds, state["confidence_thresholds"], rtol=0, atol=0
    )
    _assert_state_unchanged(state, before)


@pytest.mark.parametrize("n_layers", [5, 9])
def test_legacy_attention_names(matcher_module, checkpoint9, monkeypatch, n_layers):
    state = _clone_state(checkpoint9)
    for i in range(9):
        for attention in ("self_attn", "cross_attn"):
            old = f"transformers.{i}.{attention}."
            new = f"{attention}.{i}."
            state = {name.replace(old, new): value for name, value in state.items()}
    before = _clone_state(state)
    _download_fixture(monkeypatch, state)

    model = matcher_module.LightGlue(
        features="raco-aliked", n_layers=n_layers, **MODEL_CONF
    )

    _assert_parameters_loaded(model, checkpoint9)
    torch.testing.assert_close(
        model.confidence_thresholds, _expected_thresholds(n_layers)
    )
    _assert_state_unchanged(state, before)


@pytest.mark.parametrize(
    "bad_key", ["input_proj.weight", "transformers.0.self_attn.Wqkv.weight"]
)
def test_other_parameter_shape_mismatches_still_raise(
    matcher_module, checkpoint9, monkeypatch, bad_key
):
    state = _clone_state(checkpoint9)
    state[bad_key] = state[bad_key][:-1].clone()
    before = _clone_state(state)
    _download_fixture(monkeypatch, state)

    with pytest.raises(RuntimeError, match=f"size mismatch for {re.escape(bad_key)}"):
        matcher_module.LightGlue(features="raco-aliked", n_layers=5, **MODEL_CONF)

    _assert_state_unchanged(state, before)


def test_reduced_depth_threshold_controls_stopping_and_pruning(
    matcher_module, checkpoint9, monkeypatch
):
    _download_fixture(monkeypatch, _clone_state(checkpoint9))
    model = matcher_module.LightGlue(features="raco-aliked", n_layers=5, **MODEL_CONF)
    confidence = torch.tensor([[0.85]])

    # At layer 1, five-layer threshold ~0.84493; nine-layer threshold ~0.86412.
    # Truncating the checkpoint vector would invert both decisions below.
    assert bool(model.check_if_stop(confidence, confidence, 1, 2))
    keep = model.get_pruning_mask(confidence, torch.zeros_like(confidence), 1)
    assert not bool(keep.item())


@pytest.mark.parametrize("n_layers", [1, 5])
@pytest.mark.parametrize("adaptive", [False, True])
def test_real_cpu_forward(matcher_module, checkpoint9, monkeypatch, n_layers, adaptive):
    _download_fixture(monkeypatch, _clone_state(checkpoint9))
    model = matcher_module.LightGlue(
        features="raco-aliked",
        n_layers=n_layers,
        depth_confidence=0.95 if adaptive else -1,
        width_confidence=0.99 if adaptive else -1,
        **MODEL_CONF,
    ).eval()
    generator = torch.Generator(device="cpu").manual_seed(183)
    data = {}
    for name, points in (("image0", 8), ("image1", 7)):
        data[name] = {
            "keypoints": torch.rand(1, points, 2, generator=generator)
            * torch.tensor([64.0, 48.0]),
            "descriptors": torch.randn(
                1, points, MODEL_CONF["input_dim"], generator=generator
            ),
            "image_size": torch.tensor([[64.0, 48.0]]),
        }

    with torch.no_grad():
        result = model(data)

    assert 1 <= result["stop"] <= n_layers
    if not adaptive:
        assert result["stop"] == n_layers
    for suffix, points, other_points in (("0", 8, 7), ("1", 7, 8)):
        matches = result[f"matches{suffix}"]
        scores = result[f"matching_scores{suffix}"]
        assert matches.shape == scores.shape == (1, points)
        assert matches.dtype == torch.long
        assert matches.device.type == scores.device.type == "cpu"
        assert bool(((matches >= -1) & (matches < other_points)).all())
        assert bool(torch.isfinite(scores).all())
        assert bool(((scores >= 0) & (scores <= 1)).all())
        assert result[f"prune{suffix}"].shape == (1, points)
