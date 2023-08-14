from typing import List, Optional, Union

import cv2
import numpy as np
import torch


def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str,
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(
    path: str,
    grayscale: bool = False,
    resize: int = None,
    fn: str = "max",
    interp: str = "area",
) -> torch.Tensor:
    img = read_image(path, grayscale=grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    return numpy_image_to_torch(img), torch.Tensor(scales)


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to grayscale."""
    scale = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    image = (image * scale).sum(-3, keepdim=True)
    return image


def match_pair(extractor, matcher, image0, image1, scales0=None, scales1=None):
    device = image0.device
    data = {"image0": image0[None].cuda(), "image1": image1[None].cuda()}
    img0, img1 = data["image0"], data["image1"]
    feats0, feats1 = extractor({"image": img0}), extractor({"image": img1})
    pred = {
        **{k + "0": v for k, v in feats0.items()},
        **{k + "1": v for k, v in feats1.items()},
        **data,
    }
    pred = {**pred, **matcher(pred)}
    pred = {
        k: v.to(device).detach()[0] if isinstance(v, torch.Tensor) else v
        for k, v in pred.items()
    }
    if scales0 is not None:
        pred["keypoints0"] = (pred["keypoints0"] + 0.5) / scales0[None] - 0.5
    if scales1 is not None:
        pred["keypoints1"] = (pred["keypoints1"] + 0.5) / scales1[None] - 0.5
    del feats0, feats1
    torch.cuda.empty_cache()

    # create match indices
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    matches0, mscores0 = pred["matches0"], pred["matching_scores0"]
    valid = matches0 > -1
    matches = torch.stack([torch.where(valid)[0], matches0[valid]], -1)
    return {**pred, "matches": matches, "matching_scores": mscores0[valid]}
