# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import onnxruntime as ort
from typing import List

from onnx_runner import load_image , rgb_to_grayscale , viz2d


class LightGlueOnnxRunner():
    def __init__(self , lightglue_path: str , extractor_path=None,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],):
        self.extractor = (
            ort.InferenceSession(extractor_path, providers=providers)
            if extractor_path is not None else None
        )
        
        self.lightglue = ort.InferenceSession(lightglue_path, providers=providers)

        # Check for invalid models.
        lightglue_inputs = [i.name for i in self.lightglue.get_inputs()]
        if self.extractor is not None and "image0" in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is end-to-end. Please do not pass the extractor_path argument."
            )
        elif self.extractor is None and "image0" not in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is not end-to-end. Please pass the extractor_path argument."
            )

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1):
        if self.extractor is None:
            inference_start_time = time.time()
            
            kpts0, kpts1, matches0, matches1, mscores0, mscores1 = self.lightglue.run(
                None,
                {
                    "image0": image0,
                    "image1": image1,
                },
            )
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            infence_end_time = time.time()
            
            print(f"Only inference takes time : {round(infence_end_time - inference_start_time , 4)} s")
            return m_kpts0, m_kpts1
        else:
            inference_start_time = time.time()
            
            kpts0, scores0, desc0 = self.extractor.run(None, {"image": image0})
            kpts1, scores1, desc1 = self.extractor.run(None, {"image": image1})

            matches0, matches1, mscores0, mscores1 = self.lightglue.run(
                None,
                {
                    "kpts0": self.normalize_keypoints(
                        kpts0, image0.shape[2], image0.shape[3]),
                    "kpts1": self.normalize_keypoints(
                        kpts1, image1.shape[2], image1.shape[3]),
                    "desc0": desc0,
                    "desc1": desc1,
                },
            )
            infence_end_time = time.time()
            
            print(f"Only inference takes time : {round(infence_end_time - inference_start_time , 4)} s")
            
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            
            return m_kpts0, m_kpts1

    @staticmethod
    def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)

    @staticmethod
    def post_process(kpts0, kpts1, matches0, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        # create match indices
        valid = matches0[0] > -1
        matches = np.stack([np.where(valid)[0], matches0[0][valid]], -1)
        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1

def main():
    parser = argparse.ArgumentParser(description='Run LightGlue onnx demo.')
    parser.add_argument('--inputdir0', type=str , required=True , help='xxxx')
    parser.add_argument('--inputdir1', type=str , help='xxxx')
    parser.add_argument(
        "--lightglue-path", type=str, required=True,
        help="Path to the LightGlue ONNX model or end-to-end LightGlue pipeline.",
    )
    parser.add_argument('--extractor-type', type=str , default="SuperPoint" , 
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.")
    parser.add_argument(
        "--extractor-path" , type=str , default=None , required=False,
        help="Path to the feature extractor ONNX model. If this argument is not provided, it is assumed that lightglue_path refers to an end-to-end model.",
    )
    parser.add_argument(
        "--img-size" , nargs="+" , type=int , default=512 , required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the images to this value. Otherwise, please provide two integers (height width) to resize both images to this size, or four integers (height width height width).",
    )
    parser.add_argument(
        "--trt",action="store_true",
        help="Whether to use TensorRT (experimental). Note that the ONNX model must NOT be exported with --mp or --flash.",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Whether to visualize the results."
    )
    parser.add_argument('--savedir', type=str , help='xxxx')

    args = parser.parse_args()
    
    input_path0 = args.inputdir0 if args.inputdir0 is not None else r"data/dir0"
    input_path1 = args.inputdir1 if args.inputdir1 is not None else r"data/dir1"
    save_dir = args.savedir if args.savedir is not None else r"data/output"
    img_size = args.img_size
    if isinstance(img_size, List):
        if len(img_size) == 1:
            size0 = size1 = img_size[0]
        elif len(img_size) == 2:
            size0 = size1 = img_size
        elif len(img_size) == 4:
            size0, size1 = img_size[:2], img_size[2:]
        else:
            raise ValueError("Invalid img_size. Please provide 1, 2, or 4 integers.")
    else:
        size0 = size1 = img_size
    
    
    image0 ,scales0 = load_image(input_path0 , resize=size0)
    image1, scales1 = load_image(input_path1, resize=size1)

    extractor_type = args.extractor_type.lower()
    if extractor_type == "superpoint":
        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)
    elif extractor_type == "disk":
        pass
    else:
        raise NotImplementedError(
            f"Unsupported feature extractor type: {extractor_type}."
        )
    
    # Load ONNX models
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if args.trt:
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/cache",
                },
            )
        ] + providers

    lg_feature_matcher = LightGlueOnnxRunner(
        extractor_path=args.extractor_path,
        lightglue_path=args.lightglue_path,
        providers=providers
    )
    
    # Inference
    start_time = time.time()
    m_kpts0, m_kpts1 = lg_feature_matcher.run(image0, image1, scales0, scales1)
    end_time = time.time()
    print(f"LightGlueOnnxRunner whole process takes time : {round(end_time - start_time , 4)} s")
    
    
    if args.viz:
        orig_image0, _ = load_image(input_path0)
        orig_image1, _ = load_image(input_path1)
        viz2d.plot_images(
            [orig_image0[0].transpose(1, 2, 0), orig_image1[0].transpose(1, 2, 0)]
        )
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.plt.show()

        
    
    
if __name__ == '__main__':
    main()