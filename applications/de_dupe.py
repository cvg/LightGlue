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

logger = loguru.logger
logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="DEBUG")


sys.path.append("/datadrive/codes/opensource/features/LightGlue")


from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot



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



def main():
    img1_path = "/datadrive/codes/retail/cvtoolkit/download/OSA/Baby - Pampers/Pampers - Diapers/1927cf96a103029a908322d19cd10746.jpg"
    img0_path = "/datadrive/codes/retail/cvtoolkit/download/OSA/Baby - Pampers/Pampers - Diapers/d74409e54a50968b81d0048919e39113.jpg"
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    match_pair(img0, img1)
    

if __name__ == "__main__":
    main()