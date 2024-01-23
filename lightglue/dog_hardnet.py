import warnings

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_grayscale
from kornia.feature import HardNet, LAFDescriptor, laf_from_center_scale_ori
from packaging import version

from .utils import Extractor
from .sift import SIFT

try:
    import pycolmap
except ImportError:
    pycolmap = None


class DoGHardNet(SIFT):
    default_conf = {
        "nms_radius": 0,  # None to disable filtering entirely.
        "max_num_keypoints": 2048,
        "backend": "opencv",  # in {opencv, pycolmap, pycolmap_cpu, pycolmap_cuda}
        "detection_threshold": -1,  # from COLMAP
        "edge_threshold": -1,
        "first_octave": -1,  # only used by pycolmap, the default of COLMAP
        "num_octaves": 4,
        "force_num_keypoints": True,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        super()._init(conf)
        self.laf_desc = LAFDescriptor(HardNet(True)).eval()


    def _forward(self, data: dict) -> dict:
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        device = image.device
        self.laf_desc = self.laf_desc.to(device)
        self.laf_desc.descriptor = self.laf_desc.descriptor.eval()
        pred = []
        im_size = data.get("image_size").long()
        for k in range(len(image)):
            img = image[k]
            if im_size is not None:
                w, h = data["image_size"][k]
                img = img[:, : h.to(torch.int32), : w.to(torch.int32)]
            p = self.extract_single_image(img)
            lafs = laf_from_center_scale_ori(
                p["keypoints"].reshape(1, -1, 2),
                6.0 * p["scales"].reshape(1, -1, 1, 1),
                torch.rad2deg(p["oris"]).reshape(1, -1, 1),
            ).to(device)
            p["descriptors"] = self.laf_desc(img[None], lafs).reshape(-1, 128)
            pred.append(p)
        pred = {k: torch.stack([p[k] for p in pred], 0).to(device) for k in pred[0]}
        return pred



        