import pytest
import torch
import torch.nn.functional as F

from lightglue.aliked import DKD


@pytest.mark.parametrize(
    "sub_pixel, dtype",
    [(False, torch.float32), (True, torch.float32), (True, torch.float64)],
)
@pytest.mark.parametrize(
    "image_sizes", [None, [(20, 16)], [(12, 10)], [(12, 10), (15, 12)]]
)
def test_padding_keeps_score_map_coordinates(sub_pixel, dtype, image_sizes):
    batch_size = len(image_sizes) if image_sizes is not None else 1
    scores = torch.zeros(batch_size, 1, 16, 20, dtype=dtype)
    expected_points = [(6, 5), (9, 7)][:batch_size]
    for i, (x, y) in enumerate(expected_points):
        scores[i, 0, y, x] = 1
        if image_sizes is not None and image_sizes[i] != (20, 16):
            # A stronger peak in the padding must not be returned.
            scores[i, 0, 12, 16] = 2
    image_size = None if image_sizes is None else torch.tensor(image_sizes).float()

    keypoints, keypoint_scores, _ = DKD(radius=2)(
        scores, sub_pixel=sub_pixel, image_size=image_size
    )

    for points, values, expected in zip(keypoints, keypoint_scores, expected_points):
        assert points.shape == (1, 2)
        pixels = (points + 1) * scores.new_tensor([19, 15]) / 2
        torch.testing.assert_close(pixels, scores.new_tensor([expected]))
        torch.testing.assert_close(values, scores.new_ones(1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_padding_preserves_subpixel_values_and_gradients(dtype):
    scores = torch.zeros(1, 1, 10, 12, dtype=dtype)
    scores[0, 0, 5, 6] = 0.9
    scores[0, 0, 5, 7] = 0.8
    scores[0, 0, 4, 6] = 0.7
    scores.requires_grad_()
    padded = F.pad(scores, (0, 8, 0, 6))

    expected_points, expected_scores, expected_dispersion = DKD(radius=2)(scores)
    points, values, dispersion = DKD(radius=2)(
        padded, image_size=torch.tensor([[12.0, 10.0]])
    )
    expected_pixels = (expected_points[0] + 1) * scores.new_tensor([11, 9]) / 2
    pixels = (points[0] + 1) * scores.new_tensor([19, 15]) / 2
    torch.testing.assert_close(pixels, expected_pixels)
    torch.testing.assert_close(values[0], expected_scores[0])
    torch.testing.assert_close(dispersion[0], expected_dispersion[0])

    loss = pixels.sum() + values[0].sum() + dispersion[0].sum()
    expected_loss = (
        expected_pixels.sum() + expected_scores[0].sum() + expected_dispersion[0].sum()
    )
    gradient = torch.autograd.grad(loss, scores)[0]
    expected_gradient = torch.autograd.grad(expected_loss, scores)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    torch.testing.assert_close(gradient, expected_gradient)
