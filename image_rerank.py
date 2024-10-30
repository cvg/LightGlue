from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import numpy as np
import cv2
import json
import time
import os
import yaml
import tqdm
import sys
import pandas as pd
import loguru
import requests

logger = loguru.logger
logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")

sys.path.append("/datadrive/codes/retail/delfino")

import common.feature_factory as ff
import common.engine_factory as ef


yaml_file = "/datadrive/codes/retail/delfino/configs/retrieval_dev_test.yaml"
with open(yaml_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# print(config)

img_global_feature_config = config["Features"]["image_global"]
img_global_feature = ff.create_feature(img_global_feature_config)
# print("img_global_feature: ", type(img_global_feature), img_global_feature)


# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher


def calculate_similarity(topk_img, warped_query_img):
    similarity = []
    for i in range(len(topk_img)):
        if topk_img[i] == "" or warped_query_img[i] == "":
            similarity.append(0)
            continue
    
        feat0 = img_global_feature.extract(image_url=warped_query_img[i])
        feat1 = img_global_feature.extract(image_url=topk_img[i])
        feat0 = np.array(feat0).reshape((1, len(feat0)))
        feat1 = np.array(feat1).reshape((1, len(feat1)))
        sim = np.dot(feat0, feat1.T)
        similarity.append(sim[0][0])
    return similarity


def get_topk(topn_label, k):
    topk_label = []
    for i in range(len(topn_label)):
        if topn_label[i] in topk_label:
            continue
        topk_label.append(topn_label[i])
        if len(topk_label) == k:
            break
    if len(topk_label) < k:
        #extend topk_label to k with -1
        topk_label.extend(["-1"] * (k - len(topk_label)))
        
    return topk_label


def download_with_retry(index, image_url, img_folder, num_retries=3):
    retry = 0
    # print("image_url: ", image_url, type(image_url))
    while retry < num_retries:
        try:
            image_path = os.path.join(img_folder, f"{index}_{image_url.split('/')[-1]}.jpg")
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                return image_path
            res = requests.get(image_url)
            if res.status_code != 200:
                raise ValueError(f"Error from {image_url}")
            image = np.array(bytearray(res.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image)
            return image_path
        except Exception as e:
            print("e: ", e)
            logger.warning(f"Failed to download the image from {image_url} {retry} times")
        retry += 1
    
    logger.error(f"Failed to download the image from {image_url} after {num_retries} times")
    return None


def main():
    csv_file = "/datadrive/codes/opensource/features/LightGlue/assets/10k/100_url.csv"
    img_folder = "/datadrive/codes/opensource/features/LightGlue/assets/10k/imgs"
    aligned_folder = "/datadrive/codes/opensource/features/LightGlue/assets/10k/aligned"    
    df0 = pd.read_csv(csv_file)
    os.makedirs(aligned_folder, exist_ok=True)
    
    t_total = 0
    k = 5
    m = 93
    
    df_target = df0.copy()
    df_target = df0.iloc[m:m+1].copy()
    rerank_label_list = []
    rerank_score_list = []
    # print("df_target: ", df_target)
    
    for index, row in tqdm.tqdm(df_target.iterrows()): 
        t0 = time.time()
        p_id = row['ProductId']
        topn_label = eval(row['topn_label'])
        topn_url = eval(row['top10_URL'])
        crop_url = row['CropUrl']
        print("ProductId: ", p_id, "ml label: ", topn_label[0], "original res: ", topn_label[0]==p_id)
        # print("topn_label: ", type(topn_label))
        
        # get topk labels
        topk_label = get_topk(topn_label, k)
        topk_label = [int(x) for x in topk_label]
        topk_url = topn_url[0:k]
        print("topk_label: ", topk_label)
        # print("topk_url: ", topk_url)
        
        # download crop images
        topk_crop_paths = []
        i = 0
        for i in range(len(topk_url)):
            if topk_url[i] == "":
                topk_crop_paths.append("")
                continue
            crop_path = download_with_retry(topk_label[i], topk_url[i], img_folder, 5)
            topk_crop_paths.append(crop_path)
        # print("topk_crop_paths: ", topk_crop_paths)
        
        # download query image
        query_path = download_with_retry(p_id, crop_url, img_folder, 5)
        # print("query_path: ", query_path)
        
        # The critical step.
        # For each image in galley, first warp the query image to the same perspective as the gallery image.
        # Then calculate the similarity between the warped query image and the gallery image.
        # return the max similarity and the corresponding label.
        warped_query_path = []
        i = 0
        for i in range(len(topk_crop_paths)):
            query_img = query_path
            gallery_img = topk_crop_paths[i]
            if gallery_img == "":
                warped_query_path.append("")
                continue
            
            img_path = os.path.join(aligned_folder, f"{index}_{i}_{p_id}_{topk_label[i]}.jpg")
            # if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            #     warped_query_path.append(img_path)
            #     continue
            
            image0 = load_image(gallery_img).cuda()
            image1 = load_image(query_img).cuda()
            # print("image0: ", image0.shape)
            # print("image1: ", image1.shape)

            extractor = extractor_aliked
            matcher = matcher_aliked
            
            # extract local features
            feats0 = extractor.extract(image0)
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
            # matched_count = src_pts.shape[0]
            # if matched_count < 40:
            #     warped_query_path.append("")
            #     continue
            
            if src_pts.shape[0] < 4:
                warped_query_path.append("")
                continue
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                warped_query_path.append("")
                continue
            
            matchesMask = mask.ravel().tolist()
            # print("matchesMask: ", matchesMask)
            # calculate the number of inliers
            inliers = 0
            inlier_src_pts = []
            inlier_dst_pts = []
            j = 0
            for j in range(len(matchesMask)):
                if matchesMask[j] == 1:
                    inliers += 1
                    inlier_src_pts.append(src_pts[j])
                    inlier_dst_pts.append(dst_pts[j])
            print("inliers: ", inliers, inliers/len(matchesMask))
            if inliers < 20:
                warped_query_path.append("")
                continue

            #stitching  
            img0 = cv2.imread(gallery_img) 
            img1 = cv2.imread(query_img)  
            h, w = img0.shape[:2]  
            # print(h, w, M)
            img_aligned = cv2.warpPerspective(img1, M, (w, h))
            cv2.imwrite(img_path, img_aligned)
            warped_query_path.append(img_path)
            
        # print("warped_query_path: ", warped_query_path)
        
        # calculate similarity
        query_img_list = [query_path] * len(topk_crop_paths)
        sim = calculate_similarity(topk_crop_paths, warped_query_path)
        # print("sim: ", sim)
        max_sim = np.max(sim)
        max_index = np.argmax(sim)
        if max_sim < 0.6:
            rerank_label = topk_label[0]
        else:
            rerank_label = topk_label[max_index]
        print("max: ", sim, max_sim, max_index)
        print("productId: ", p_id, "final label: ", rerank_label, "rerank res: ", rerank_label==p_id)
        
        rerank_label_list.append(rerank_label)
        rerank_score_list.append(max_sim)
        
        t1 = time.time()
        t_total += (t1 - t0)
    
    df_target['rerank_label'] = rerank_label_list
    df_target['rerank_score'] = rerank_score_list
    df_target.to_csv("assets/10k/rerank_result.csv", index=False)
    print("AVG time: ", t_total/len(df_target))
        
        
if __name__ == "__main__":
    main()
    exit(0)        
    
