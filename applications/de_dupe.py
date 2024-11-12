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
import glob

logger = loguru.logger
logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="DEBUG")


sys.path.append("/datadrive/codes/opensource/features/LightGlue")
sys.path.append("/datadrive/codes/retail/delfino")

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot

from utils.download_crops import download_image_with_retry
from utils.faster_infer import TritonClientGRPC, UnitDetector


TRT_URL = os.environ.get("TRT_URL", "localhost:8001")


def filter_points(src_points:np.ndarray, dst_points, img_h, img_w)->np.ndarray:
    """
    This function will take the points and filter out the points according to some criteria.
    """
    total_points = src_points.shape[0]
    
    # Criteria 1: constrain the src_points to be with [0.7, 1]*width
    x_threshold = int(0.6*img_w)
    # print("x_threshold: ", x_threshold, src_points.shape)
    cond = (src_points[:, 0] > x_threshold) & (src_points[:, 0] < img_w)
    dst_points = dst_points[cond]
    src_points = src_points[cond]
    # print("dst_points: ", dst_points.shape)
    # print("src_points: ", src_points.shape)
    
    
    # Criteria 1: constrain the dst_points to be with [0.7, 1]*width
    x_threshold = int(0.4*img_w)
    # print("x_threshold: ", x_threshold, dst_points.shape)
    cond = (dst_points[:, 0] < img_w) & (dst_points[:, 0] < x_threshold)
    dst_points = dst_points[cond]
    src_points = src_points[cond]
    # print("dst_points: ", dst_points.shape)
    # print("src_points: ", src_points.shape)
    
    # # Criteria 3: filter out the outliers after clustering the slope between matched points
    # slopes = (src_points[:,1] - dst_points[:, 1]) / (src_points[:, 0] - dst_points[:, 0])
    # slopes = slopes.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=2, n_init=10).fit(slopes)

    # # Get labels and cluster centroids  
    # labels = kmeans.labels_  
    # # centroids = kmeans.cluster_centers_  
    
    # # Find the centroid index of the largest cluster (which is the main cluster)  
    # counts = np.bincount(labels)  
    # main_cluster_index = np.argmax(counts) 
    # print("main_cluster_index: ", main_cluster_index)
    
    # cond = labels == main_cluster_index
    # src_points = src_points[cond]
    # dst_points = dst_points[cond]
    # print("dst_points: ", dst_points.shape)
    # print("src_points: ", src_points.shape)

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
    
    # img_tmp = img0.copy() # DEBUG
    img_w_ori = img0.shape[1]
    img_h_ori = img0.shape[0]
    size = (480, 640)
    scale_x = img_w_ori/size[0] # img_h/size_w
    scale_y = img_h_ori/size[1] # img_w/size_h
    print("scale_x: ", scale_x, "scale_y: ", scale_y)
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
    points0 = features0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = features1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    t1 = time.time()
    print("feature & match time: ", t1-t0)

    # plot and save the matches
    points0 = points0.cpu().numpy()
    points1 = points1.cpu().numpy()
    
    # plot_images([img0, img1])
    # print("points0: ", points0.shape)
    # # plot_keypoints(points0)
    # # plot_keypoints(points1)
    # plot_matches(points0, points1, color='lime', lw=1)
    # save_plot("origin_matches.png")
    
    # filter the points
    points0, points1 = filter_points(points0, points1, img0.shape[0], img0.shape[1])
    
    # plot_images([img0, img1])
    # print("points0: ", points0.shape)
    # # plot_keypoints(points0)
    # # plot_keypoints(points1)
    # plot_matches(points0, points1, color='lime', lw=1)
    # save_plot("matches.png")
    
    if points0.shape[0] < 4:
        return np.eye(3), points0, points1
    
    try:
        # find homography
        H, mask = cv2.findHomography(points0, points1, cv2.RANSAC, 3.0, maxIters=2000)
        # print("Homography: ", H)
        # print("Inliers: ", np.sum(mask))
    
        # find Fundamental matrix
        F, _ = cv2.findFundamentalMat(points0, points1, cv2.FM_RANSAC, 5.0, 0.99)
        # print("Fundamental Matrix: ", F)
        
        # draw the overlapping areas
        # Apply the homography to the source image  
        # warped_image = cv2.warpPerspective(img0, H, (img1.shape[1], img1.shape[0]))
        # print("warped_image: ", warped_image.shape)
        # cv2.imwrite("warped_image_0.png", warped_image)
        
        # contours, _ = cv2.findContours(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img1 = img1[..., ::-1]
        # img_draw = img1.copy()
        # cv2.fillPoly(img_draw, contours[0], (0, 255, 0))
        # cv2.fillPoly(img_draw, contours, (0, 255, 0))
        
        # # merge the images
        # alpha = 0.7
        # beta = 1 - alpha
        # img_draw = cv2.addWeighted(img1, alpha, img_draw, beta, 0)
        # # Display the image  
        # cv2.imwrite("overlapping_area_0.png", img_draw)
        
        # Apply the homography to the destination image
        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        scale_matrix_inv = np.linalg.inv(scale_matrix) # origin -> resized
        H_scaled = np.dot(H, scale_matrix_inv)
        H_scaled = np.dot(scale_matrix, H_scaled)
        # warped_image = cv2.warpPerspective(img_tmp, H_scaled, (img_w_ori, img_h_ori))

    except Exception as e:
        print("Error: ", e)
        return np.eye(3)
    
    return H_scaled, points0, points1



def get_homographies(img_series:np.ndarray, output_dir:str):
    """
    This function will take a series of images and return the overlapping areas.
    """
    count = img_series.shape[0]
    print("count: ", count)
    
    Hs = []
    for i in tqdm.tqdm(range(count-1)):
        img0 = img_series[i]
        img1 = img_series[i+1]
        H, pts0, pts1 = match_pair(img0, img1)
        Hs.append(H)
        
        # DEBUG
        print("points0: ", pts0.shape)
        if pts0.shape[0] < 1:
            continue
        plot_images([cv2.resize(img0, (480, 640)), cv2.resize(img1, (480, 640))])
        # plot_keypoints(points0)
        # plot_keypoints(points1)
        plot_matches(pts0, pts1, color='lime', lw=1)
        save_plot(f"{output_dir}/matches_{i}_{i+1}.png")
        print("-------------------------------------------------")
    
    Hs.append(np.eye(3))
    return Hs



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



def sort_key_func(item):  
    base_name = os.path.basename(item)  # get file name with extension  
    num = os.path.splitext(base_name)[0]  # remove extension  
    return int(num)  



def iou(box1, box2):
    """
    This function will calculate the intersection over union of two boxes.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    if x5 > x6 or y5 > y6:
        return 0
    intersection = (x6 - x5) * (y6 - y5)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection / union


def iof(box1, box2):
    """
    This function will calculate the intersection over foreground of two boxes.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    if x5 > x6 or y5 > y6:
        return 0
    intersection = (x6 - x5) * (y6 - y5)
    foreground = (x2 - x1) * (y2 - y1)
    return intersection / foreground



def detect_units(img_series:list, model_name:str, model_version:str):
    grpc_client = TritonClientGRPC(TRT_URL)
    unit_detector = UnitDetector(grpc_client, model_name, model_version)
    img_boxes = []
    for img in img_series:
        try:
            h,w,c = img.shape
            boxes = unit_detector.detect(img)
            boxes = [[box[0]*w, box[1]*h, box[2]*w, box[3]*h] for box in boxes]
            img_boxes.append(boxes)
        except Exception as e:
            print("Error: ", e)
            img_boxes.append([])
    return img_boxes



def dedupe_units(img_series:list, Hs:list, img_boxes:list):
    box_delete_flags = []
    box_delete_flags.append([0]*len(img_boxes[0]))
    for i in range(1, len(img_series)):
        delete_flag = [0]*len(img_boxes[i])
        box_remained = []
        H_prev = Hs[i-1]
        img_boxes_prev = img_boxes[i-1]
        img_boxes_curr = img_boxes[i]
        
        if len(img_boxes_prev) == 0 or len(img_boxes_curr) == 0:
            box_delete_flags.append(delete_flag)
            print("No boxes in the images.")
            continue
        
        if np.array_equal(H_prev, np.eye(3)):
            box_delete_flags.append(delete_flag)
            print(f"No transformation between {i-1} and {i}")
            continue
        
        # get the warped image and the contour
        img_warped = cv2.warpPerspective(img_series[i-1], H_prev, (img_series[i].shape[1], img_series[i].shape[0]))
        cv2.imwrite("img_warped.png", img_warped)
        contours, _ = cv2.findContours(cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img1 = img_series[i].copy()
        img_draw = img1.copy()
        # cv2.fillPoly(img_draw, contours[0], (0, 255, 0))
        cv2.fillPoly(img_draw, contours, (0, 255, 0))
        
        # merge the images
        alpha = 0.7
        beta = 1 - alpha
        img_draw = cv2.addWeighted(img1, alpha, img_draw, beta, 0)
        
        # Display the image  
        cv2.imwrite("overlapping_area_1.png", img_draw)
        
        if len(contours) == 0:
            continue
        elif len(contours) == 1:
            contour = contours[0]
        else:
            max_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    contour = c
        
        # get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # draw the bounding box
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite("overlapping_area_2.png", img_draw)
        
        # get the boxes in the current image
        box_to_dedupe = []
        for k, box in enumerate(img_boxes_curr):
            x1, y1, x2, y2 = box
            img_countor_iof = iof([x1, y1, x2, y2], [x, y, x+w, y+h])
            if img_countor_iof > 0.99:
                delete_flag[k] = 1
                continue
            elif img_countor_iof < 0.01:
                delete_flag[k] = 0
            else:
                delete_flag[k] = 1
                cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                box_to_dedupe.append(i)
        cv2.imwrite("overlapping_area_3.png", img_draw)
        print("box_to_dedupe: ", len(box_to_dedupe), "box_remained: ", len(box_remained))
        
        warped_boxes = []
        for k in box_to_dedupe:
            x1, y1, x2, y2 = img_boxes_curr[k]
            H_pre_inv = np.linalg.inv(H_prev)
            warped_box = cv2.perspectiveTransform(np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32), H_pre_inv)
            warped_boxes.append(warped_box)
        
        # get the intersection over union of the boxes in the previous image
        img0_tmp = img_series[i-1].copy()
        for box_idx, warped_box in zip(box_to_dedupe, warped_boxes):
            x1, y1 = warped_box[0][0]
            x2, y2 = warped_box[0][2]
            # print("x1, y1, x2, y2: ", x1, y1, x2, y2)
            cv2.rectangle(img0_tmp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            max_iou = 0
            for box in img_boxes_prev:
                iou_score = iou([x1, y1, x2, y2], box)
                if iou_score > max_iou:
                    max_iou = iou_score
                if max_iou > 0.5:
                    break
            # print("max_iou: ", max_iou)
            if max_iou < 0.5:
                cv2.rectangle(img0_tmp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5)
                delete_flag[box_idx] = 0
        
        box_delete_flags.append(delete_flag)
        cv2.imwrite("overlapping_area_4.png", img0_tmp)
        print("Original boxes: ", len(img_boxes_curr), "Deleted boxes: ", sum(delete_flag))
        
    return box_delete_flags
            


def process_image_folder(img_dir:str, model_name:str, model_version:str):
    """
    This function will read the images from the folder and process them.
    """
    # read the images
    img_series = []
    img_paths = []
    img_paths = glob.glob(img_dir + "/*.jpg")
    img_paths = sorted(img_paths, key=sort_key_func)
    print("img_paths: ", img_paths)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_series.append(img)
        
    output_dir = os.path.join(img_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # get the homographies
    Hs = get_homographies(np.array(img_series), output_dir)
    # print("Hs: ", Hs)
    
    # detect the units
    img_boxes = detect_units(img_series, model_name, model_version)
    
    # dedupe the units
    box_delete_flags = dedupe_units(img_series, Hs, img_boxes)
    print("box_delete_flags: ", len(box_delete_flags))
    
    # draw the boxes
    for i, img in enumerate(img_series):
        delete_flags = box_delete_flags[i]
        boxes = img_boxes[i]
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if delete_flags[j] == 1:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            else:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
            cv2.imwrite(f"{output_dir}/{i}.jpg", img)
    
    return (img_series, Hs, box_delete_flags)



def main():
    csv_file = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/OSA_Original_Image.csv"
    img_dir = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images"
    # prepare_input(csv_file, img_dir)
    
    # img0_path = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images/2488/Home cleaning/Mr Clean/42.jpg"
    # img1_path = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images/2488/Home cleaning/Mr Clean/43.jpg"
    # img0 = cv2.imread(img0_path)
    # img1 = cv2.imread(img1_path)
    # match_pair(img0, img1)
    
    img_dir = "/datadrive/codes/opensource/features/LightGlue/data/dedupe/osa_images/2488/Oral Care/Crest"
    model_name = "unit_hpc_yolo_v5"
    model_version = "20230107"
    process_image_folder(img_dir, model_name, model_version)



if __name__ == "__main__":
    main()