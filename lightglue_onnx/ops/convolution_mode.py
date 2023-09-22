import torch


def patch_disk_convolution_mode(extractor: torch.nn.Module) -> None:
    """Fix failed ONNX export for DISK extractor due to aten::_convolution_mode.

    See https://github.com/pytorch/pytorch/issues/68880.
    """
    for m in extractor.modules():
        if (
            isinstance(m, torch.nn.Conv2d)
            and m.padding_mode == "zeros"
            and m.padding == "same"
        ):
            m.padding = tuple(k // 2 for k in m.kernel_size)
