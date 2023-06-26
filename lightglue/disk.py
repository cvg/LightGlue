import torch
import torch.nn as nn
import kornia
from types import SimpleNamespace


class DISK(nn.Module):
    default_conf = {
        'weights': 'depth',
        'max_num_keypoints': None,
        'desc_dim': 128,
        'nms_window_size': 5,
        'detection_threshold': 0.0,
        'pad_if_not_divisible': True,
    }
    required_data_keys = ['image']

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)
        self.model = kornia.feature.DISK.from_pretrained(self.conf.weights)

    def forward(self, data: dict) -> dict:
        image = data['image']

        features = self.model(
            image,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_window_size,
            score_threshold=self.conf.detection_threshold,
            pad_if_not_divisible=self.conf.pad_if_not_divisible
        )
        keypoints = [f.keypoints for f in features]
        scores = [f.detection_scores for f in features]
        descriptors = [f.descriptors for f in features]
        del features

        keypoints = torch.stack(keypoints, 0)
        scores = torch.stack(scores, 0)
        descriptors = torch.stack(descriptors, 0)

        return {
            'keypoints': keypoints.to(image),
            'keypoint_scores': scores.to(image),
            'descriptors': descriptors.to(image),
        }
