"""Regression coverage for OpenCV resize dimensions and reported pixel scales."""
from lightglue import utils as image_utils
import cv2
import numpy as np
import pytest

@pytest.mark.parametrize('shape, expected', [((1, 1000, 3), (1, 100, 3)), ((1000, 1, 3), (100, 1, 3)), ((1, 200, 3), (1, 100, 3))])
@pytest.mark.parametrize('interp', ['area', 'linear', 'nearest'])
def test_narrow_images_keep_nonzero_short_edge(shape, expected, interp):
    module = image_utils
    image = np.full(shape, 128, np.uint8)
    resized, scale = module.resize_image(image, 100, interp=interp)
    assert resized.shape == expected
    np.testing.assert_array_equal(resized, np.full(expected, 128, np.uint8))
    assert scale == (expected[1] / shape[1], expected[0] / shape[0])

@pytest.mark.parametrize('target', [0, -1])
def test_nonpositive_scalar_target_is_rejected(target):
    with pytest.raises(ValueError, match='positive'):
        image_utils.resize_image(np.zeros((5, 5, 3), np.uint8), target)

@pytest.mark.parametrize('target, expected', [(5, (2, 5)), ((4, 7), (4, 7)), ([6, 9], (6, 9))])
def test_regular_resize_matches_opencv(target, expected):
    image = np.arange(4 * 10 * 3, dtype=np.uint8).reshape(4, 10, 3)
    out, scale = image_utils.resize_image(image, target)
    np.testing.assert_array_equal(out, cv2.resize(image, (expected[1], expected[0]), interpolation=cv2.INTER_AREA))
    assert scale == (expected[1] / 10, expected[0] / 4)

def test_narrow_grayscale_is_supported():
    out, scale = image_utils.resize_image(np.zeros((1, 1000), np.uint8), 100)
    assert out.shape == (1, 100)
    assert scale == (0.1, 1.0)

def test_load_image_uses_fixed_resize_path(tmp_path):
    path = tmp_path / 'narrow.png'
    assert cv2.imwrite(str(path), np.full((1, 1000, 3), 255, np.uint8))
    tensor = image_utils.load_image(path, resize=100)
    assert tuple(tensor.shape) == (3, 1, 100)
    assert bool((tensor == 1).all())
