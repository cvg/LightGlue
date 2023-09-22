from types import SimpleNamespace
from typing import Optional, Tuple

import kornia
import torch
import torch.nn.functional as F


def nms(
    signal: torch.Tensor, window_size: int = 5, cutoff: float = 0.0
) -> torch.Tensor:
    if window_size % 2 != 1:
        raise ValueError(f"window_size has to be odd, got {window_size}")

    _, ixs = F.max_pool2d(
        signal,
        kernel_size=window_size,
        stride=1,
        padding=window_size // 2,
        return_indices=True,
    )

    _, _, h, w = signal.shape
    coords = torch.arange(h * w, device=signal.device).reshape(h, w)
    nms = ixs == coords

    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)


def heatmap_to_keypoints(
    heatmap: torch.Tensor,
    n: Optional[int] = None,
    window_size: int = 5,
    score_threshold: float = 0.0,
):
    """Inference-time nms-based detection protocol."""
    nmsed = nms(heatmap, window_size=window_size, cutoff=score_threshold)

    bcyx = nmsed.nonzero()
    xy = bcyx[..., 2:].flip((1,))
    scores = heatmap[nmsed]

    if n is not None:
        kpts_len = torch.tensor(scores.shape[0])  # Still dynamic despite trace warning
        max_keypoints = torch.minimum(torch.tensor(n), kpts_len)
        scores, indices = torch.topk(scores, max_keypoints, dim=0)
        return xy[indices], scores

    return xy, scores


class DISK(torch.nn.Module):
    default_conf = {
        "weights": "depth",
        "max_num_keypoints": None,
        "desc_dim": 128,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }
    required_data_keys = ["image"]

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)
        self.model = kornia.feature.DISK.from_pretrained(self.conf.weights)

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # image.shape == (1, 3, H, W)
        if self.conf.pad_if_not_divisible:
            h, w = image.shape[2:]
            pd_h = (16 - h % 16) % 16
            pd_w = (16 - w % 16) % 16
            image = torch.nn.functional.pad(image, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descriptors = self.model.heatmap_and_dense_descriptors(image)

        if self.conf.pad_if_not_divisible:
            heatmaps = heatmaps[..., :h, :w]
            descriptors = descriptors[..., :h, :w]

        # heatmaps.shape == (1, 1, H, W), descriptors.shape == (1, desc_dim, H, W)

        keypoints, scores = heatmap_to_keypoints(
            heatmaps,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_window_size,
            score_threshold=self.conf.detection_threshold,
        )

        # keypoints.shape == (N, 2), scores.shape == (N,)

        descriptors = descriptors[..., keypoints.T[1], keypoints.T[0]].permute(0, 2, 1)
        descriptors = F.normalize(descriptors, dim=-1)

        # descriptors.shape == (1, N, desc_dim)

        # Insert artificial batch dimension
        return keypoints[None], scores[None], descriptors
