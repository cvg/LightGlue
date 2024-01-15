import warnings

import cv2
import numpy as np
import torch
from packaging import version
from kornia.color import rgb_to_grayscale
from kornia.feature import laf_from_center_scale_ori, LAFDescriptor, HardNet
from .utils import Extractor

try:
    import pycolmap
except ImportError:
    pycolmap = None


def filter_dog_point(points, scales, angles, image_shape, nms_radius, scores=None):
    if len(scales) == 0:
        keep = [True for x in range(len(scales))]
        return np.array(keep).astype(bool)
    h, w = image_shape
    ij = np.round(points - 0.5).astype(int).T[::-1]

    # Remove duplicate points (identical coordinates).
    # Pick highest scale or score
    s = scales if scores is None else scores
    buffer = np.zeros((h, w))
    np.maximum.at(buffer, tuple(ij), s)
    keep = np.where(buffer[tuple(ij)] == s)[0]

    # Pick lowest angle (arbitrary).
    ij = ij[:, keep]
    buffer[:] = np.inf
    o_abs = np.abs(angles[keep])
    np.minimum.at(buffer, tuple(ij), o_abs)
    mask = buffer[tuple(ij)] == o_abs
    ij = ij[:, mask]
    keep = keep[mask]

    if nms_radius > 0:
        # Apply NMS on the remaining points
        buffer[:] = 0
        buffer[tuple(ij)] = s[keep]  # scores or scale

        local_max = torch.nn.functional.max_pool2d(
            torch.from_numpy(buffer).unsqueeze(0),
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(0)
        is_local_max = buffer == local_max.numpy()
        keep = keep[is_local_max[tuple(ij)]]
    return keep


def run_opencv_sift(features: cv2.Feature2D, image: np.ndarray) -> np.ndarray:
    """
    Detect keypoints using OpenCV Detector.
    Optionally, perform description.
    Args:
        features: OpenCV based keypoints detector and descriptor
        image: Grayscale image of uint8 data type
    Returns:
        points: 2D array of keypoints
        scales: 1D array of keypoint scales
        angles: 1D array of keypoint orientations in radians
        scores: 1D array of responses
    """
    detections = features.detect(image, None)
    points = np.array([k.pt for k in detections], dtype=np.float32)
    scores = np.array([k.response for k in detections], dtype=np.float32)
    scales = np.array([k.size for k in detections], dtype=np.float32)
    angles = np.deg2rad(np.array([k.angle for k in detections], dtype=np.float32))
    return points, scores, scales, angles


class DoGHardNet(Extractor):
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
        backend = self.conf.backend
        if backend.startswith("pycolmap"):
            if pycolmap is None:
                raise ImportError(
                    "Cannot find module pycolmap: install it with pip"
                    "or use backend=opencv."
                )
            options = {
                "peak_threshold": self.conf.detection_threshold,
                "edge_threshold": self.conf.edge_threshold,
                "first_octave": self.conf.first_octave,
                "num_octaves": self.conf.num_octaves,
                "normalization": pycolmap.Normalization.L2,  # L1_ROOT is buggy.
            }
            device = (
                "auto" if backend == "pycolmap" else backend.replace("pycolmap_", "")
            )
            if (
                backend == "pycolmap_cpu" or not pycolmap.has_cuda
            ) and pycolmap.__version__ < "0.5.0":
                warnings.warn(
                    "The pycolmap CPU SIFT is buggy in version < 0.5.0, "
                    "consider upgrading pycolmap or use the CUDA version.",
                    stacklevel=1,
                )
            else:
                options["max_num_features"] = self.conf.max_num_keypoints
            self.sift = pycolmap.Sift(options=options, device=device)
        elif backend == "opencv":
            self.sift = cv2.SIFT_create(
                contrastThreshold=self.conf.detection_threshold,
                nfeatures=self.conf.max_num_keypoints,
                edgeThreshold=self.conf.edge_threshold,
                nOctaveLayers=self.conf.num_octaves,
            )
        else:
            backends = {"opencv", "pycolmap", "pycolmap_cpu", "pycolmap_cuda"}
            raise ValueError(
                f"Unknown backend: {backend} not in " f"{{{','.join(backends)}}}."
            )
        self.laf_desc = LAFDescriptor(HardNet(True)).eval()

    def extract_single_image(self, image: torch.Tensor):
        device = image.device
        image_np = image.cpu().numpy().squeeze(0)

        if self.conf.backend.startswith("pycolmap"):
            if version.parse(pycolmap.__version__) >= version.parse("0.5.0"):
                detections, descriptors = self.sift.extract(image_np)
                scores = None  # Scores are not exposed by COLMAP anymore.
            else:
                detections, scores, descriptors = self.sift.extract(image_np)
            keypoints = detections[:, :2]  # Keep only (x, y).
            scales, angles = detections[:, -2:].T
            if scores is not None and (
                self.conf.backend == "pycolmap_cpu" or not pycolmap.has_cuda
            ):
                # Set the scores as a combination of abs. response and scale.
                scores = np.abs(scores) * scales
        elif self.conf.backend == "opencv":
            # TODO: Check if opencv keypoints are already in corner convention
            keypoints, scores, scales, angles = run_opencv_sift(
                self.sift, (image_np * 255.0).astype(np.uint8)
            )

        if len(keypoints) == 0:
            warnings.warn("No keypoints detected")
            keypoints = np.zeros((1, 2)).astype(np.float32)
            scales = np.ones((1)).astype(np.float32)
            angles = np.zeros((1)).astype(np.float32)
            scores = np.zeros((1)).astype(np.float32)

        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": angles,
            "descriptors": descriptors,
        }
        if scores is not None:
            pred["keypoint_scores"] = scores

        # sometimes pycolmap returns points outside the image. We remove them
        if self.conf.backend.startswith("pycolmap"):
            is_inside = (
                pred["keypoints"] + 0.5 < np.array([image_np.shape[-2:][::-1]])
            ).all(-1)
            pred = {k: v[is_inside] for k, v in pred.items()}

        if (self.conf.nms_radius is not None) and len(pred["scales"] > 0):
            keep = filter_dog_point(
                pred["keypoints"],
                pred["scales"],
                pred["oris"],
                image_np.shape,
                self.conf.nms_radius,
                pred["keypoint_scores"],
            )
            pred2 = {k: v[keep] for k, v in pred.items()}
            if pred2['keypoints'] is not None:
                pred = pred2
        pred = {k: torch.from_numpy(v).float() for k, v in pred.items()}
        if scores is not None:
            # Keep the k keypoints with highest score
            num_points = self.conf.max_num_keypoints
            if num_points is not None and len(pred["keypoints"]) > num_points:
                indices = torch.topk(pred["keypoint_scores"], num_points).indices
                pred = {k: v[indices] for k, v in pred.items()}
        lafs = laf_from_center_scale_ori(pred["keypoints"].reshape(1, -1, 2),
                                         6.0 * pred["scales"].reshape(1, -1, 1, 1),
                                         torch.rad2deg(pred["oris"]).reshape(1, -1, 1)).to(device)
        self.laf_desc = self.laf_desc.to(device)
        self.laf_desc.descriptor = self.laf_desc.descriptor.eval()
        pred["descriptors"] = self.laf_desc(image[None], lafs).reshape(-1, 128)
        return pred

    def _forward(self, data: dict) -> dict:
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        device = image.device
        pred = []
        im_size = data.get("image_size").long()
        for k in range(len(image)):
            img = image[k]
            if im_size is not None:
                w, h = data["image_size"][k]
                img = img[:, :h.to(torch.int32), :w.to(torch.int32)]
            p = self.extract_single_image(img)
            pred.append(p)
        pred = {k: torch.stack([p[k] for p in pred], 0).to(device) for k in pred[0]}
        return pred
