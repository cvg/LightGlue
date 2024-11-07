"""
In this file, we will write the code to mark the overlapping areas of the images.
The input will be a series of images and the output will be the images with the overlapping areas marked.
"""

import numpy as np
import cv2
import json
import time
import sys
import os
import loguru
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

logger = loguru.logger
logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="DEBUG")


sys.path.append("/datadrive/codes/opensource/features/LightGlue")
sys.path.append("/datadrive/codes/retail/delfino")

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot

from utils.download_crops import download_image_with_retry


def filter_points(src_points:np.ndarray, dst_points, img_h, img_w)->np.ndarray:
    """
    This function will take the points and filter out the points according to some criteria.
    """
    total_points = src_points.shape[0]
    
    # Criteria 1: constrain the src_points to be with [0.7, 1]*width
    x_threshold = int(0.6*img_w)
    print("x_threshold: ", x_threshold, src_points.shape)
    cond = (src_points[:, 0] > x_threshold) & (src_points[:, 0] < img_w)
    dst_points = dst_points[cond]
    src_points = src_points[cond]
    print("dst_points: ", dst_points.shape)
    print("src_points: ", src_points.shape)
    
    
    # Criteria 1: constrain the dst_points to be with [0.7, 1]*width
    x_threshold = int(0.4*img_w)
    print("x_threshold: ", x_threshold, dst_points.shape)
    cond = (dst_points[:, 0] < img_w) & (dst_points[:, 0] < x_threshold)
    dst_points = dst_points[cond]
    src_points = src_points[cond]
    print("dst_points: ", dst_points.shape)
    print("src_points: ", src_points.shape)
    
    # Criteria 3: filter out the outliers after clustering the slope between matched points
    slopes = (src_points[:,1] - dst_points[:, 1]) / (src_points[:, 0] - dst_points[:, 0])
    slopes = slopes.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10).fit(slopes)

    # Get labels and cluster centroids  
    labels = kmeans.labels_  
    # centroids = kmeans.cluster_centers_  
    
    # Find the centroid index of the largest cluster (which is the main cluster)  
    counts = np.bincount(labels)  
    main_cluster_index = np.argmax(counts) 
    print("main_cluster_index: ", main_cluster_index)
    
    cond = labels == main_cluster_index
    src_points = src_points[cond]
    dst_points = dst_points[cond]
    print("dst_points: ", dst_points.shape)
    print("src_points: ", src_points.shape)

    valid_points = src_points.shape[0]
    print("Total points: ", total_points, "Valid points: ", valid_points, "Percentage: ", valid_points/total_points)
    return src_points, dst_points


def match_pair(img0:np.ndarray, img1:np.ndarray) -> np.ndarray:
    """
    This function will take two images and return the overlapping areas.
    """
    # ALIKED+LightGlue
    extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher

    size = (480, 640)
    img0 = cv2.resize(img0, size)
    img1 = cv2.resize(img1, size)
    img0 = img0[..., ::-1]
    img1 = img1[..., ::-1]
    t_img0 = numpy_image_to_torch(img0).cuda()
    t_img1 = numpy_image_to_torch(img1).cuda()

    t0 = time.time()
    # extract local features
    features0 = extractor_aliked.extract(t_img0)  # auto-resize the image, disable with resize=None
    features1 = extractor_aliked.extract(t_img1)

    # match the features
    matches01 = matcher_aliked({'image0': features0, 'image1': features1})
    features0, features1, matches01 = [rbd(x) for x in [features0, features1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    print("matches: ", matches)
    points0 = features0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = features1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    t1 = time.time()
    print("time: ", t1-t0)

    # plot and save the matches
    points0 = points0.cpu().numpy()
    points1 = points1.cpu().numpy()
    
    plot_images([img0, img1])
    print("points0: ", points0.shape)
    # plot_keypoints(points0)
    # plot_keypoints(points1)
    plot_matches(points0, points1, color='lime', lw=1)
    save_plot("origin_matches.png")
    
    # filter the points
    points0, points1 = filter_points(points0, points1, img0.shape[0], img0.shape[1])
    
    plot_images([img0, img1])
    print("points0: ", points0.shape)
    # plot_keypoints(points0)
    # plot_keypoints(points1)
    plot_matches(points0, points1, color='lime', lw=1)
    save_plot("matches.png")
    
    # find homography
    H, _ = cv2.findHomography(points0, points1, cv2.RANSAC, 5.0, maxIters=10000)
    print("Homography: ", H)
    
    # find Fundamental matrix
    F, _ = cv2.findFundamentalMat(points0, points1, cv2.FM_RANSAC, 5.0, 0.99)
    print("Fundamental Matrix: ", F)
    
    # draw the overlapping areas
    # Apply the homography to the source image  
    warped_image = cv2.warpPerspective(img0, H, (img1.shape[1], img1.shape[0]))
    print("warped_image: ", warped_image.shape)
    cv2.imwrite("warped_image.png", warped_image)
    
    contours, _ = cv2.findContours(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img1 = img1[..., ::-1]
    img_draw = img1.copy()
    # cv2.fillPoly(img_draw, contours[0], (0, 255, 0))
    cv2.fillPoly(img_draw, contours, (0, 255, 0))
    
    # merge the images
    alpha = 0.7
    beta = 1 - alpha
    img_draw = cv2.addWeighted(img1, alpha, img_draw, beta, 0)
    
    # Display the image  
    cv2.imwrite("overlapping_area.png", img_draw)
    
    return points0, points1


def prepare_input(csv_file:str, img_dir:str):
    """
    This function will read the csv file and prepare the input for the match_pair function.
    """
    df_input = pd.read_csv(csv_file)
    print("df_input: ", df_input.head())
    os.makedirs(img_dir, exist_ok=True)
    
    for idx, row in tqdm.tqdm(df_input.iterrows()):
        try:
            folder = img_dir
            store_number = str(row["StoreNum"])
            
            folder = os.path.join(folder, store_number)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            
            task_group = row['TaskGroupName']
            task_group.strip()
            folder = os.path.join(folder, task_group)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            
            task_name = row['TaskName']
            task_name.strip()
            folder = os.path.join(folder, task_name)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
                
            img_num = row['Image number']
            img_name = f"{folder}/{img_num}.jpg"
            
            if os.path.exists(img_name):
                continue
            img_url = row['Image URL']
            _, img = download_image_with_retry(img_url)
            cv2.imwrite(img_name, img)
        except Exception as e:
            print("Error: ", e)
            continue


def main():
    csv_file = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/OSA_Original_Image.csv"
    img_dir = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images"
    # prepare_input(csv_file, img_dir)
    
    img0_path = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images/2488/Home cleaning/Mr Clean/42.jpg"
    img1_path = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images/2488/Home cleaning/Mr Clean/43.jpg"
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    match_pair(img0, img1)

if __name__ == "__main__":
    main()