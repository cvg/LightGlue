from types import SimpleNamespace

import kornia
import torch
import torch.nn as nn

from .utils import ImagePreprocessor


class DISK(nn.Module):
    default_conf = {
        "weights": "depth",
        "max_num_keypoints": None,
        "desc_dim": 128,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }

    preprocess_conf = {
        **ImagePreprocessor.default_conf,
        "resize": 1024,
        "grayscale": False,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)
        self.model = kornia.feature.DISK.from_pretrained(self.conf.weights)

    def forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        features = self.model(
            image,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_window_size,
            score_threshold=self.conf.detection_threshold,
            pad_if_not_divisible=self.conf.pad_if_not_divisible,
        )
        keypoints = [f.keypoints for f in features]
        scores = [f.detection_scores for f in features]
        descriptors = [f.descriptors for f in features]
        del features

        keypoints = torch.stack(keypoints, 0)
        scores = torch.stack(scores, 0)
        descriptors = torch.stack(descriptors, 0)

        return {
            "keypoints": keypoints.to(image).contiguous(),
            "keypoint_scores": scores.to(image).contiguous(),
            "descriptors": descriptors.to(image).contiguous(),
        }

    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats
