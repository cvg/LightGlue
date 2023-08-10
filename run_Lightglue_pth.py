# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
from typing import Optional , Tuple , Union , List

from lightglue import LightGlue , SuperPoint , DISK
from lightglue.utils import load_image , rbd
from lightglue import viz2d

class LightGlueFeatureMatcher():
    """
        Use SuperPoint features combined with LightGlue.
    """
    def __init__(self , withline=False , device="cpu") -> None:
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)
        self.withline = withline
        self.device = device
    
    def _preprocess(self , image , size: Union[List[int], int], fn: str='max' , \
                    interp: Optional[str]='area') -> Tuple[np.ndarray , int]:
        """
            Resize an image to a fixed size, or according to max or min edge.
        """
        h , w  = image[:2]
        fn = {'max': max, 'min': min}[fn]
        if isinstance(size, int):
            scale = size / fn(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
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
        
        resize_img = cv2.resize(image , (w_new , h_new) , interpolation=mode)
        
        return resize_img , scale 
    
    def _numpy2torch(self , image:np.ndarray) -> torch.Tensor:
        """
            Normalize the image tensor and reorder the dimensions.
        """
        if image.ndim == 3:
            image = image.transpose((2,0,1)) # HxWxC => CxHxW
        elif image.ndim == 2:
            image = image[None]
        else:
            raise ValueError(f"Shape {image.shape} does not match the input format ")
        
        tensor = torch.tensor(image / 255 , dtype=torch.float)
        
        return tensor
    
    def _remove_batch_dim(self , data:dict) -> dict:
        """
            Remove batch dimension from elements in data
        """
        return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()}
    
    def _inference(self , image0:torch.Tensor , image1:torch.Tensor) \
            ->Tuple[torch.Tensor , torch.Tensor , dict]:
        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))
        matches01 = self.matcher({
            'image0' : feats0 , 
            'image1' : feats1
        })
   
        return feats0 , feats1 , matches01
    
    def _postprocess(self , srcImage0 , srcImage1 , feats0 , feats1 , matches01) \
            -> Tuple[np.array , np.array , None]:
        feats0 , feats1 , matches01 = [
            self._remove_batch_dim(x) for x in [feats0 , feats1 , matches01]]
        
        kpts0 , kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0 , m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        axes = viz2d.plot_images([srcImage0 , srcImage1])
        viz2d.plot_matches(m_kpts0 , m_kpts1 , color='lime' , lw=0.2)
        viz2d.add_text(0 , f'Stop after {matches01["stop"]} layers', fs=20)
   
        kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
        if self.withline:
            return kpc0 , kpc1 , axes
        
        viz2d.plot_images([srcImage0, srcImage1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        
        return kpc0 , kpc1 , axes
    
    def _inference_img(self , srcImage0 , srcImage1 , resize=None):
        if resize is not None:
            image0 = self._preprocess(srcImage0)
            image1 = self._preprocess(srcImage1)
        else:
            image0 = srcImage0.copy()
            image1 = srcImage1.copy()
            
        image0_tensor = self._numpy2torch(image0)
        image1_tensor = self._numpy2torch(image1)
        
        inference_start_time = time.time()
        feats0 , feats1 , matches01 = self._inference(image0_tensor , image1_tensor)
        infence_end_time = time.time()
        print(f"Only inference takes time : {round(infence_end_time - inference_start_time , 4)} s")
        
        kpc0, kpc1 , axes = self._postprocess(srcImage0 , srcImage1 , feats0 , feats1 , matches01)
        
        return kpc0 , kpc1 , axes
    
    def _save_plot(self , path , axes=None):
        viz2d.save_plot(path)
        

def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """
        Read an image from path as RGB or grayscale
    """
    if not Path(path).exists():
        raise FileNotFoundError(f'No image at path {path}.')
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description='Run LightGlue demo.')
    parser.add_argument('--inputdir0', type=str , help='xxxx')
    parser.add_argument('--inputdir1', type=str , help='xxxx')
    parser.add_argument('--savedir', type=str , help='xxxx')
    parser.add_argument('--withline', action="store_true" , default=False , help='xxxx')

    args = parser.parse_args()
        
    input_dir0 = args.inputdir0 if args.inputdir0 is not None else r"data/dir0"
    input_dir1 = args.inputdir1 if args.inputdir1 is not None else r"data/dir1"
    save_dir = args.savedir if args.savedir is not None else r"data/output"
    withline = args.withline
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()} , chose device {device_}")
    
    if not (osp.exists(input_dir0) and osp.exists(input_dir1)) :
        raise FileNotFoundError(f"Path `{input_dir0}` or `{input_dir1}` is not exists")
    image_dir0 = Path(input_dir0)
    image_dir1 = Path(input_dir1)
    
    lg_feature_matcher = LightGlueFeatureMatcher(withline , device_)
    
    for path0 , path1 in zip(os.listdir(image_dir0) , os.listdir(image_dir1)):
        try:
            print("# --------------------------- #")
            image_path0 = osp.join(image_dir0 , path0)
            print(f"Image0 Path : {image_path0}")
            image_path1 = osp.join(image_dir1 , path1)
            print(f"Image1 Path : {image_path1}")
            
            image0 = read_image(image_path0)
            image1 = read_image(image_path1)
            
            start_time = time.time()
            kpc0, kpc1 , _ = lg_feature_matcher._inference_img(image0 , image1)
            end_time = time.time()
            
            save_name = osp.basename(path0).split(".")[0] + "_"+ osp.basename(path1).split(".")[0]
            if withline:
                save_name += "_withline.png"
            else:
                save_name += ".png"
            save_path = osp.join(save_dir , save_name)
            lg_feature_matcher._save_plot(save_path)
            
            print(f"LightGlue Feature Matcher whole process takes time : {round(end_time - start_time , 4)} s")
            print(f"Save at : {save_path}")
            
        except Exception as e:
            print(e)
        
if __name__ == '__main__':
    main()