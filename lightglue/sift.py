import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn as nn
from scipy.spatial import KDTree

from .utils import ImagePreprocessor

EPS = 1e-6


def sift_to_rootsift(x: np.ndarray) -> np.ndarray:
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


# from OpenGlue
def nms_keypoints(kpts: np.ndarray, responses: np.ndarray, radius: float) -> np.ndarray:
    # TODO: port to GPU (scatter -> SP nms -> gather)
    kd_tree = KDTree(kpts)

    sorted_idx = np.argsort(-responses)
    kpts_to_keep_idx = []
    removed_idx = set()

    for idx in sorted_idx:
        # skip point if it was already removed
        if idx in removed_idx:
            continue

        kpts_to_keep_idx.append(idx)
        point = kpts[idx]
        neighbors = kd_tree.query_ball_point(point, r=radius)
        # Variable `neighbors` contains the `point` itself
        removed_idx.update(neighbors)

    mask = np.zeros((kpts.shape[0],), dtype=bool)
    mask[kpts_to_keep_idx] = True
    return mask


def detect_kpts_opencv(features: cv2.Feature2D, image: np.ndarray) -> np.ndarray:
    """
    Detect keypoints using OpenCV Detector.
    Optionally, perform description.
    Args:
        features: OpenCV based keypoints detector and descriptor
        image: Grayscale image of uint8 data type
    Returns:
        keypoints: 1D array of detected cv2.KeyPoint
        scores: 1D array of responses
        descriptors: 1D array of descriptors
    """
    kpts, descriptors = features.detectAndCompute(image, None)
    kpts = np.array(kpts)

    responses = np.array([k.response for k in kpts], dtype=np.float32)

    # select all
    pts = np.array([k.pt for k in kpts], dtype=np.float32)
    scales = np.array([k.size for k in kpts], dtype=np.float32)
    angles = np.deg2rad(np.array([k.angle for k in kpts], dtype=np.float32))
    spts = np.concatenate([pts, scales[..., None], angles[..., None]], -1)
    return spts, responses, descriptors


class SIFT(nn.Module):
    default_conf = {
        "rootsift": True,
        "nms_radius": None,
        "max_num_keypoints": 4096,
        "detector": "opencv",  # ['pycolmap', 'pycolmap_cpu', 'pycolmap_cuda', opencv']
        "detection_threshold": 0.0066667,  # from COLMAP
        "edge_threshold": 10,
        "first_octave": -1,  # only used by pycolmap, the default of COLMAP
        "num_octaves": 4,
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
        self.sift = None  # lazy loading

    @torch.no_grad()
    def extract_cpu(self, image: torch.Tensor):
        image_np = image.cpu().numpy()[0]
        assert image.shape[0] == 1
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS

        detector = str(self.conf.detector)

        if self.sift is None and detector.startswith("pycolmap"):
            options = {
                "peak_threshold": self.conf.detection_threshold,
                "edge_threshold": self.conf.edge_threshold,
                "first_octave": self.conf.first_octave,
                "num_octaves": self.conf.num_octaves,
            }
            device = (
                "auto" if detector == "pycolmap" else detector.replace("pycolmap_", "")
            )
            if (
                detector == "pycolmap_cpu" or not pycolmap.has_cuda
            ) and pycolmap.__version__ < "0.5.0":
                warnings.warn(
                    "The pycolmap CPU SIFT is buggy in version < 0.5.0, "
                    "consider upgrading pycolmap or use the CUDA version.",
                    stacklevel=1,
                )
            else:
                options["max_num_features"] = self.conf.max_num_keypoints
            if self.conf.rootsift == "rootsift":
                options["normalization"] = pycolmap.Normalization.L1_ROOT
            else:
                options["normalization"] = pycolmap.Normalization.L2
            self.sift = pycolmap.Sift(options=options, device=device)
        elif self.sift is None and self.conf.detector == "opencv":
            self.sift = cv2.SIFT_create(
                contrastThreshold=self.conf.detection_threshold,
                nfeatures=self.conf.max_num_keypoints,
                edgeThreshold=self.conf.edge_threshold,
                nOctaveLayers=self.conf.num_octaves,
            )
        elif self.sift is None:
            raise ValueError(
                f"Unknown SIFT detector {self.conf.detector}. "
                + "Choose from (pycolmap, pycolmap_cuda, pycolmap_cpu, opencv)."
            )

        if detector.startswith("pycolmap"):
            keypoints, scores, descriptors = self.sift.extract(image_np)
            if (
                detector == "pycolmap_cpu" or not pycolmap.has_cuda
            ) and pycolmap.__version__ < "0.5.0":
                scores = (
                    np.abs(scores) * keypoints[:, 2]
                )  # set score as a combination of abs. response and scale
        elif detector == "opencv":
            # TODO: Check if opencv keypoints are already in corner convention
            keypoints, scores, descriptors = detect_kpts_opencv(
                self.sift, (image_np * 255.0).astype(np.uint8)
            )

        if self.conf.nms_radius is not None:
            mask = nms_keypoints(keypoints[:, :2], scores, self.conf.nms_radius)
            keypoints = keypoints[mask]
            scores = scores[mask]
            descriptors = descriptors[mask]

        scales = keypoints[:, 2]
        oris = keypoints[:, 3]

        if self.conf.rootsift:
            descriptors = sift_to_rootsift(descriptors)
        descriptors = torch.from_numpy(descriptors)
        keypoints = torch.from_numpy(keypoints[:, :2])  # keep only x, y
        scales = torch.from_numpy(scales)
        oris = torch.from_numpy(oris)
        scores = torch.from_numpy(scores)

        # Keep the k keypoints with highest score
        max_kps = self.conf.max_num_keypoints

        if max_kps is not None and max_kps > 0:
            max_kps = min(self.conf.max_num_keypoints, keypoints.shape[-2])
            indices = torch.topk(scores, max_kps).indices
            keypoints = keypoints[indices]
            scales = scales[indices]
            oris = oris[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]

        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": oris,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }

        return pred

    @torch.no_grad()
    def forward(self, data: dict) -> dict:
        pred = {
            "keypoints": [],
            "scales": [],
            "oris": [],
            "keypoint_scores": [],
            "descriptors": [],
        }

        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True).cpu()

        for k in range(image.shape[0]):
            img = image[k]
            if "image_size" in data.keys():
                # avoid extracting points in padded areas
                w, h = data["image_size"][k]
                img = img[:, :h, :w]
            p = self.extract_cpu(img)
            for k, v in p.items():
                pred[k].append(v)

        pred = {
            k: torch.stack(pred[k], 0).to(device=data["image"].device)
            for k in pred.keys()
        }

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
