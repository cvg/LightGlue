import kornia
import torch
import torch.nn as nn

from types import SimpleNamespace

from .utils import ImagePreprocessor


class SIFTKornia(nn.Module):
    default_conf = {
        "max_num_keypoints": -1,
        "detection_threshold": None,
        "rootsift": True,
    }

    preprocess_conf = {
        **ImagePreprocessor.default_conf,
        "resize": 1024,
        "grayscale": False,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf):
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        conf = self.conf = SimpleNamespace(**self.conf)
        self.sift = kornia.feature.SIFTFeature(
            num_features=self.conf.max_num_keypoints, rootsift=self.conf.rootsift
        )

    def forward(self, data):
        lafs, scores, descriptors = self.sift(data["image"])
        keypoints = kornia.feature.get_laf_center(lafs)
        scales = kornia.feature.get_laf_scale(lafs)
        oris = kornia.feature.get_laf_orientation(lafs)
        pred = {
            "keypoints": keypoints,  # @TODO: confirm keypoints are in corner convention
            "scales": scales,
            "oris": oris,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }

        pred["scales"] = pred["scales"]
        pred["oris"] = torch.deg2rad(pred["oris"])
        return pred

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
