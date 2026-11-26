import cv2
import numpy as np
import pytest
import torch

from lightglue import SIFT, LightGlue
from lightglue.sift import run_opencv_sift


def test_opencv_empty_feature_shapes():
    detector = cv2.SIFT_create()
    image = np.zeros((64, 96), dtype=np.uint8)
    points, scores, scales, angles, descriptors = run_opencv_sift(detector, image)

    assert points.shape == (0, 2)
    assert descriptors.shape == (0, detector.descriptorSize())
    for values in (points, scores, scales, angles, descriptors):
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float32
    assert scores.shape == scales.shape == angles.shape == (0,)


@pytest.mark.parametrize("nms_radius", [None, 0, 2])
@pytest.mark.parametrize("rootsift", [False, True])
@pytest.mark.parametrize("channels, origin", [(1, "center"), (3, "corner")])
def test_extract_blank_image(nms_radius, rootsift, channels, origin):
    extractor = SIFT(nms_radius=nms_radius, rootsift=rootsift, origin=origin)
    features = extractor.extract(torch.zeros(channels, 64, 96), resize=64)

    expected_shapes = {
        "keypoints": (1, 0, 2),
        "descriptors": (1, 0, 128),
        "keypoint_scores": (1, 0),
        "scales": (1, 0),
        "oris": (1, 0),
    }
    for name, shape in expected_shapes.items():
        assert features[name].shape == shape
        assert features[name].dtype == torch.float32
        assert features[name].device.type == "cpu"
    torch.testing.assert_close(features["image_size"], torch.tensor([[96.0, 64.0]]))


def test_opencv_nonempty_features_are_unchanged():
    image = np.random.default_rng(0).integers(0, 256, (96, 128), dtype=np.uint8)
    detector = cv2.SIFT_create()
    detections, expected_descriptors = detector.detectAndCompute(image, None)
    assert len(detections) > 0

    points, scores, scales, angles, descriptors = run_opencv_sift(detector, image)

    np.testing.assert_array_equal(
        points, np.array([k.pt for k in detections], np.float32)
    )
    np.testing.assert_array_equal(scores, [k.response for k in detections])
    np.testing.assert_array_equal(scales, [k.size for k in detections])
    np.testing.assert_array_equal(
        angles, np.deg2rad(np.array([k.angle for k in detections], np.float32))
    )
    np.testing.assert_array_equal(descriptors, expected_descriptors)


@pytest.mark.parametrize("empty0, empty1", [(True, False), (False, True), (True, True)])
def test_empty_sift_features_reach_matcher(empty0, empty1):
    extractor = SIFT(max_num_keypoints=32)
    blank = torch.zeros(1, 96, 128)
    texture = torch.rand((1, 96, 128), generator=torch.Generator().manual_seed(0))
    feats0 = extractor.extract(blank if empty0 else texture, resize=None)
    feats1 = extractor.extract(blank if empty1 else texture, resize=None)
    count0, count1 = feats0["keypoints"].shape[1], feats1["keypoints"].shape[1]
    assert (count0 == 0) == empty0
    assert (count1 == 0) == empty1

    matcher = LightGlue(
        features=None,
        input_dim=128,
        descriptor_dim=32,
        n_layers=2,
        add_scale_ori=True,
        flash=False,
    ).eval()
    with torch.no_grad():
        matches = matcher({"image0": feats0, "image1": feats1})

    assert matches["matches"].shape == (1, 0, 2)
    assert matches["scores"].shape == (1, 0)
    for index, count in [(0, count0), (1, count1)]:
        assert matches[f"matches{index}"].shape == (1, count)
        assert torch.all(matches[f"matches{index}"] == -1)
        assert torch.all(matches[f"matching_scores{index}"] == 0)
