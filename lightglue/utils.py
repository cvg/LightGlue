from pathlib import Path
import torch
import cv2
import numpy as np
from typing import Union, List, Optional, Callable
import collections.abc as collections


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = 'cpu',
                    non_blocking: bool = True):
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()
    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()}


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
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
        raise ValueError(f'Not an image: {image.shape}')
    return torch.tensor(image / 255., dtype=torch.float)


def resize_image(image: np.ndarray, size: Union[List[int], int],
                 fn: str, interp: Optional[str] = 'area') -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {'max': max, 'min': min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(path: Path, grayscale: bool = False, resize: int = None,
               fn: str = 'max', interp: str = 'area') -> torch.Tensor:
    img = read_image(path, grayscale=grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    return numpy_image_to_torch(img), torch.Tensor(scales)


def match_pair(extractor, matcher, image0, image1, scales0=None, scales1=None):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    data0, data1 = {'image': image0[None]}, {'image': image1[None]}
    feats0, feats1 = extractor(data0), extractor(data1)
    data = {'image0': {**feats0, **data0}, 'image1': {**feats1, **data1}}
    matches01 = batch_to_device(matcher(data))
    feats0, feats1 = batch_to_device(feats0), batch_to_device(feats1)
    # remove batch dim
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    if scales0 is not None:
        feats0['keypoints'] = (feats0['keypoints'] + .5) / scales0[None] - .5
    if scales1 is not None:
        feats1['keypoints'] = (feats1['keypoints'] + .5) / scales1[None] - .5
    return feats0, feats1, matches01
