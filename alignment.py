from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import numpy as np
import cv2
import json
import time
import os
import yaml
import sys

sys.path.append("/datadrive/codes/retail/delfino")

import common.feature_factory as ff
import common.engine_factory as ef


yaml_file = "/datadrive/codes/retail/delfino/configs/retrieval_dev_test.yaml"
with open(yaml_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# print(config)

img_global_feature_config = config["Features"]["image_global"]
img_global_feature = ff.create_feature(img_global_feature_config)
print("img_global_feature: ", type(img_global_feature), img_global_feature)


# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher

# SIFT+LightGlue
extractor_sift = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_sift = LightGlue(features='sift').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
query_img = "/datadrive/codes/opensource/features/LightGlue/assets/retail/q7.jpeg"
gallery_imgs = ["/datadrive/codes/opensource/features/LightGlue/assets/retail/4362623.png",
                "/datadrive/codes/opensource/features/LightGlue/assets/retail/4363150.jpeg",
                "/datadrive/codes/opensource/features/LightGlue/assets/retail/4384724.jpeg",]

similarity_list = []

for idx, gallery_img in enumerate(gallery_imgs):
    image0 = load_image(gallery_img).cuda()
    image1 = load_image(query_img).cuda()
    print("image0: ", image0.shape)
    print("image1: ", image1.shape)

    extractor = extractor_aliked
    matcher = matcher_aliked

    print("extractor ============> ", type(extractor).__name__)

    t0 = time.time()
    # extract local features
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    arr_points0 = points0.cpu().numpy()
    arr_points1 = points1.cpu().numpy()

    #find homography
    src_pts = np.float32(arr_points1).reshape(-1, 1, 2)
    dst_pts = np.float32(arr_points0).reshape(-1, 1, 2)
    # print("src_pts: ", src_pts.shape)
    # print("dst_pts: ", dst_pts.shape)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # print("matchesMask: ", matchesMask)
    # calculate the number of inliers
    inliers = 0
    inlier_src_pts = []
    inlier_dst_pts = []
    for i in range(len(matchesMask)):
        if matchesMask[i] == 1:
            inliers += 1
            inlier_src_pts.append(src_pts[i])
            inlier_dst_pts.append(dst_pts[i])
    print("inliers: ", inliers, inliers/len(matchesMask))

    #stitching  
    img0 = cv2.imread(gallery_img) 
    img1 = cv2.imread(query_img)  
    h, w = img0.shape[:2]  

    img_aligned = cv2.warpPerspective(img1, M, (w, h))

    #save result
    cv2.imwrite(f"output/{idx}.jpg", img_aligned)

    t1 = time.time()
    print("time: ", (t1 - t0))


    # calculate the image similarity
    feat0 = img_global_feature.extract(image_url=gallery_img)
    feat1 = img_global_feature.extract(image_url=f"output/{idx}.jpg")
    
    feat0 = np.array(feat0).reshape((1, len(feat0)))
    feat1 = np.array(feat1).reshape((1, len(feat1)))
    
    sim = np.dot(feat0, feat1.T)
    similarity_list.append(sim)
    print("sim: ", sim)
    
print("Result: ", np.argmax(similarity_list))


