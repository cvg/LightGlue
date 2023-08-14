from typing import Tuple

import torch


class LightGlueEnd2End(torch.nn.Module):
    def __init__(
        self, extractor: torch.nn.Module, lightglue: torch.nn.Module, safe=False
    ):
        super().__init__()
        self.extractor = extractor
        self.lightglue = lightglue
        self.safe = safe

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # image.shape == (B, 3, H, W)
        _, _, h0, w0 = image0.shape
        _, _, h1, w1 = image1.shape
        kpts0, scores0, desc0 = self.extractor(image0)
        kpts1, scores1, desc1 = self.extractor(image1)

        # kpts.shape == (1, N, 2), desc.shape == (1, N, desc_dim)

        if self.safe:
            # Script mode
            if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
                return (
                    kpts0,
                    kpts1,
                    torch.zeros(1, 0, dtype=torch.int64),
                    torch.zeros(1, 0, dtype=torch.int64),
                    torch.zeros(1, 0, dtype=torch.float32),
                    torch.zeros(1, 0, dtype=torch.float32),
                )

        matches0, matches1, mscores0, mscores1 = self.lightglue(
            normalize_keypoints(kpts0, h0, w0),
            normalize_keypoints(kpts1, h1, w1),
            desc0,
            desc1,
        )

        # matches.shape == (1, N) == mscores.shape

        return kpts0, kpts1, matches0, matches1, mscores0, mscores1


def normalize_keypoints(
    kpts: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    size = torch.tensor([w, h], dtype=torch.float32, device=kpts.device)
    shift = size / 2
    scale = size.max() / 2
    kpts = (kpts - shift) / scale
    return kpts
