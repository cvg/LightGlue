import pytest
import torch
import torch.nn.functional as F

from lightglue.lightglue import Attention


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mask_kind", ["partial", "empty_row", "all", "none"])
def test_fallback_attention_matches_sdpa(dtype, mask_kind):
    generator = torch.Generator().manual_seed(42)
    tensors = [
        torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
        for shape in [(2, 3, 4, 5), (2, 3, 6, 5), (2, 3, 6, 7)]
    ]
    mask = torch.ones(2, 1, 4, 6, dtype=torch.bool)
    if mask_kind == "partial":
        mask[..., 1::2] = False
    elif mask_kind == "empty_row":
        mask[..., 1::2] = False
        mask[..., 2, :] = False
    elif mask_kind == "all":
        mask[:] = False
    elif mask_kind == "none":
        mask = None
    attention = Attention(False)
    attention.has_sdp = False
    output = attention(*tensors, mask=mask)
    expected = F.scaled_dot_product_attention(*tensors, attn_mask=mask)
    torch.testing.assert_close(output, expected)
    upstream = torch.randn(output.shape, generator=generator, dtype=dtype)
    actual_grad = torch.autograd.grad(output, tensors, upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tensors, upstream)
    for actual, reference in zip(actual_grad, expected_grad):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference)


def test_masked_values_cannot_affect_fallback_attention():
    attention = Attention(False)
    attention.has_sdp = False
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[0.0, 0.0], [100.0, 0.0]]])
    v = torch.tensor([[[2.0, 3.0], [999.0, -999.0]]])
    mask = torch.tensor([[[True, False]]])
    torch.testing.assert_close(attention(q, k, v, mask), v[:, :1])
