from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import numpy as np
import cv2
import json
import time

# SuperPoint+LightGlue
extractor_sp = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_sp = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor_disk = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_disk = LightGlue(features='disk').eval().cuda()  # load the matcher

# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher

# SIFT+LightGlue
extractor_sift = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_sift = LightGlue(features='sift').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
img0_path = "/datadrive/codes/opensource/features/ALIKED/assets/test1/3.jpg"
img1_path = "/datadrive/codes/opensource/features/ALIKED/assets/test1/0.jpg"
# size = (640, 480)
image0 = load_image(img0_path).cuda()
image1 = load_image(img1_path).cuda()
print("image0: ", image0.shape)
print("image1: ", image1.shape)

extractors = [extractor_sp, extractor_disk, extractor_aliked, extractor_sift]
names = ['SuperPoint', 'DISK', 'ALIKED', 'SIFT']
matchers = [matcher_sp, matcher_disk, matcher_aliked, matcher_sift]
homo_dict = {}
errors = []

for name, extractor, matcher in zip(names, extractors, matchers):
    print("extractor ============> ", name)
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
    # print("feats0: ", feats0)
    # print("feats1: ", feats1)
    # print("matches: ", matches)
    # print("points0: ", points0.shape)
    # print('points1: ', points1.shape)

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
    img0 = cv2.imread(img0_path)  
    img1 = cv2.imread(img1_path)  
    h, w = img1.shape[:2]  
    H0 = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)  
    H1 = H0 @ M  
    Hs = [H0, H1]  
    x_min = 0  
    x_max = 0  
    y_min = 0  
    y_max = 0  
    for i, H in enumerate(Hs):  
        corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])  
        transformed_corners = cv2.perspectiveTransform(np.float32([corners]), H)  
        transformed_corners = np.int32(transformed_corners[0])  
        # print("transformed_corners:", transformed_corners)  
        x_min = min(x_min, min(transformed_corners[:, 0]))  
        x_max = max(x_max, max(transformed_corners[:, 0]))  
        y_min = min(y_min, min(transformed_corners[:, 1]))  
        y_max = max(y_max, max(transformed_corners[:, 1]))  
    
    xmin = min(0, x_min)  
    xmax = max(w, x_max)  
    ymin = min(0, y_min)  
    ymax = max(h, y_max)
    print("x_min: ", x_min, "x_max: ", x_max, "y_min: ", y_min, "y_max: ", y_max)  
    
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    for i in range(len(Hs)):  
        Hs[i] = np.dot(translation, Hs[i]) 
    
    images = [img0, img1]  
    width = xmax - xmin  
    height = ymax - ymin  
    print("width: ", width, "height: ", height)  
    pano = np.zeros((height, width, 3), np.uint8)
    
    for img, H in zip(images, Hs):
        # print("H: ", H)
        cv2.warpPerspective(img, H, (width, height), pano, borderMode=cv2.BORDER_TRANSPARENT)

    #save result
    cv2.imwrite(f"output/{name}.jpg", pano)
    homo_dict[name] = M.tolist()
    
    t1 = time.time()
    print("time: ", (t1 - t0))
    
    # NEW ref 
    ref_H_inv = np.linalg.inv(Hs[-1])
    for i in range(len(Hs)):
        Hs[i] = ref_H_inv@Hs[i]
    
    x_min = 0  
    x_max = 0  
    y_min = 0  
    y_max = 0  
    for i, H in enumerate(Hs):  
        corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])  
        transformed_corners = cv2.perspectiveTransform(np.float32([corners]), H)  
        transformed_corners = np.int32(transformed_corners[0])  
        # print("transformed_corners:", transformed_corners)  
        x_min = min(x_min, min(transformed_corners[:, 0]))  
        x_max = max(x_max, max(transformed_corners[:, 0]))  
        y_min = min(y_min, min(transformed_corners[:, 1]))  
        y_max = max(y_max, max(transformed_corners[:, 1]))  
    
    xmin = min(0, x_min)  
    xmax = max(w, x_max)  
    ymin = min(0, y_min)  
    ymax = max(h, y_max)
    print("x_min: ", x_min, "x_max: ", x_max, "y_min: ", y_min, "y_max: ", y_max)  
    
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    for i in range(len(Hs)):  
        Hs[i] = np.dot(translation, Hs[i]) 
    
    images = [img0, img1]  
    width = xmax - xmin  
    height = ymax - ymin  
    print("width: ", width, "height: ", height)  
    pano = np.zeros((height, width, 3), np.uint8)
    
    for img, H in zip(images, Hs):
        # print("H: ", H)
        cv2.warpPerspective(img, H, (width, height), pano, borderMode=cv2.BORDER_TRANSPARENT)

    #save result
    cv2.imwrite(f"output/new_ref_{name}.jpg", pano)
    homo_dict[name] = M.tolist()
    
    #calculate the reprojection error
    # Assume pts_src and pts_dst are arrays of source and destination points  
    # and H is the homography matrix  
    pts_src = np.squeeze(np.array(inlier_src_pts))
    pts_dst = np.squeeze(np.array(inlier_dst_pts))
    # pts_src = np.squeeze(arr_points1)
    # pts_dst = np.squeeze(arr_points0)

    pts_src_homogeneous = cv2.convertPointsToHomogeneous(pts_src)
    pts_src_homogeneous_reshaped = pts_src_homogeneous.reshape(-1,3)
    predicted_pts_dst_homogenous = np.matmul(M, pts_src_homogeneous_reshaped.T)
    
    # Convert homogeneous coordinates back to 2D  
    predicted_pts_dst = cv2.convertPointsFromHomogeneous(predicted_pts_dst_homogenous.T)
    # Calculate the Euclidean distance
    predicted_pts_dst = np.squeeze(predicted_pts_dst)
    reprojection_error = np.sqrt(np.sum((pts_dst - predicted_pts_dst)**2, axis=1))
    print("reprojection_error: ", np.sum(reprojection_error[0:4]))
    # for i in range(0, 1):
    #     print(pts_dst[i])
    #     print(predicted_pts_dst[i])
    #     print("reprojection_error: ", reprojection_error[i])
    # Average reprojection error  
    mean_error = np.mean(reprojection_error[0:4])
    print("mean_error: ", mean_error)
    errors.append(mean_error)
    
    # # USE ORB features and the M to calculate the reprojection error
    # orb = cv2.ORB_create()
    # kp0, des0 = orb.detectAndCompute(img0, None)
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # # print("des0: ", des0.shape)
    # # print("des1: ", des1.shape)
    # # print("kp0: ", len(kp0))
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = matcher.match(des1, des0)
    # print("matches: ", len(matches))
    
    # good = []
    # pts_src = []
    # pts_dst = []
    # for match in matches:
    #     pts_src.append(kp1[match.queryIdx].pt)
    #     pts_dst.append(kp0[match.trainIdx].pt)
        
    # pts_src = np.squeeze(np.array(pts_src))
    # pts_dst = np.squeeze(np.array(pts_dst))
    # print("pts_src: ", pts_src.shape)
    # print("pts_dst: ", pts_dst.shape)
    # pts_src_homogeneous = cv2.convertPointsToHomogeneous(pts_src)
    # pts_src_homogeneous_reshaped = pts_src_homogeneous.reshape(-1,3)
    # predicted_pts_dst_homogenous = np.matmul(M, pts_src_homogeneous_reshaped.T)
    
    # # Convert homogeneous coordinates back to 2D
    # predicted_pts_dst = cv2.convertPointsFromHomogeneous(predicted_pts_dst_homogenous.T)
    # # Calculate the Euclidean distance
    # predicted_pts_dst = np.squeeze(predicted_pts_dst)
    # reprojection_error = np.sqrt(np.sum((pts_dst - predicted_pts_dst)**2, axis=1))
    # # print("reprojection_error: ", np.sum(reprojection_error))
    # for i in range(0, 1):
    #     print(pts_dst[i])
    #     print(predicted_pts_dst[i])
    #     print("reprojection_error: ", reprojection_error[i])
    # # Average reprojection error
    # mean_error = np.mean(reprojection_error)
    # print("ORB mean_error: ", mean_error)
    

with open("output/stitch_info.json", 'w') as f:
    json.dump(homo_dict, f)
    
print("best feature: ", names[np.argmin(errors)])