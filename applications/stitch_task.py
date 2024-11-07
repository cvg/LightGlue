
import numpy as np
import cv2
import json
import time
import loguru
from concurrent.futures import ThreadPoolExecutor
import requests
import functools
from pathlib import Path
import pandas as pd
from hashlib import md5

import glob
import os
import sys

sys.path.append("/datadrive/codes/opensource/features/LightGlue")

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue import LightGlue, SIFT, ALIKED
from lightglue.utils import numpy_image_to_torch, rbd



# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher

# SIFT+LightGlue
extractor_sift = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_sift = LightGlue(features='sift').eval().cuda()  # load the matcher

# extractor_aliked = extractor_sift
# matcher_aliked = matcher_sift


logger = loguru.logger
logger.add("stitch_task.log", format="{time} {level} {message}", level="INFO", rotation="1 MB", compression="zip")

DOWNLOAD_CACHE_DIR = "/datadrive/codes/opensource/features/LightGlue/assets/imgs"
NETWORK_NUM_RETRY = 3
NETWORK_NUM_THREADS = 4

def parse_jpeg(content):
    jpeg = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    assert img is not None, 'Failed to decode JPEG'
    h, w = img.shape[:2]
    del img
    return jpeg, (w, h)


def download(url, parse_func, num_retry):
    path = Path(DOWNLOAD_CACHE_DIR) / f'{Path(url).name}'
    print("path: ", path)

    if path and path.is_file():
        print("path is file")
        return parse_func(path.read_bytes())

    ## Download
    for _ in range(num_retry):
        try:
            rsp = requests.get(url, timeout=60)
            if rsp.status_code == 200:
                result = parse_func(rsp.content)
                if path: path.write_bytes(rsp.content)
                print("write to path: ", path)
                return result
            else:
                logger.warning(f'Failed to download {url}: HTTP code = {rsp.status_code}')
        except requests.RequestException as e:
            logger.warning(f'Failed to download {url}: {e}')
    return None


def download_and_parse(urls, parse_func, num_retry, num_workers):
    with ThreadPoolExecutor(num_workers) as pool:
        result = pool.map(functools.partial(download, parse_func=parse_func, num_retry=num_retry), urls)
    result = [v for v in result if v is not None]
    return result if len(result) == len(urls) else None


def parse_req(req):
    # print(req.keys())
    is_video_stitching = req['type'] == 'VIDEO'
    num_imgs = len(req['imageUrls'])
    
    urls = []
    unsorted_names = []
    
    #url only
    for i in range(len(req['imageUrls'])):
        url = req['imageUrls'][i]
        # name = Path(url).name
        # name = name.split('.')[0]
        # unsorted_names.append(name)
        urls.append(url)
    # print("urls", urls)

    ## Download images
    logger.info('Downloading images ...')
    jpegs = download_and_parse(urls, parse_jpeg, num_retry=NETWORK_NUM_RETRY, num_workers=NETWORK_NUM_THREADS)
    assert jpegs, 'Failed to get JPEG'
    img_size = None
    imgs = []
    for i, (jpeg, size) in enumerate(jpegs):
        unsorted_names.append(md5(jpeg).hexdigest())
        img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        img_size = size
        imgs.append(img)
    # print("unsorted_names", unsorted_names)
    
    ## Get homography
    if is_video_stitching:
        stitching_info = json.loads(req['stitchingInfo'])
        homographies = stitching_info['images']
        # assert num_imgs == len(homographies), f'Invalid homography number {len(homographies)}, expected {num_imgs}'
        Hs = []
        
        for i, item in enumerate(homographies):
            name = item['md5']
            #assert name in unsorted_names, f'Invalid name {name}'
            H = item['homographyOfPano']
            assert len(H) == 9, f'Invalid homography size {len(H)}, expected 9'
            index = item['index']
            assert index == i, f'Invalid index {index}, expected {i}'
            H = np.float32(H).reshape(3, 3)
            Hs.append(H)

    else:
        # assert num_imgs == 1 + len(req['pair']), f"Invalid homography number {len(req['pair'])}, expected {num_imgs - 1}"
        Hs = [np.eye(3, 3, dtype=np.float32)]
        for i, pair in enumerate(req['pair']):
            last = pair["image1Index"] if "image1Index" in pair.keys() else 0
            cur = pair["image2Index"]
            assert last == i and last + 1 == cur, f'Invalid pair ({last}, {cur}), expected ({i}, {i + 1})'
            H = pair['homography']
            assert len(H) == 9, f'Invalid homography size {len(H)}, expected 9'
            H = np.dot(Hs[-1], np.float32(H).reshape(3, 3)) 
            Hs.append(H)
    return imgs, Hs, unsorted_names, img_size

def get_boundingBox(Hs, corners):
    ## Get offset and size
    pts = []
    for H in Hs:
        warp_corners = cv2.perspectiveTransform(corners, H)
        pts.append(warp_corners)
    pts = np.concatenate(pts)
    rect = cv2.boundingRect(pts)
    print("rect", rect)
    return rect


def verify_corners(corners):
    corners = corners.reshape(4, 2)
    _, _, w, h = cv2.boundingRect(corners)
    if min(w, h) < 1: return -1
    if not cv2.isContourConvex(corners): return -1
    if corners[0][0] >= corners[1][0]: logger.warning(f"Top left X is greater than top right X")
    if corners[3][0] >= corners[2][0]: logger.warning(f"Bottom left X is greater than bottom right X")
    if corners[0][1] >= corners[3][1]: logger.warning(f"Top left Y is greater than bottom left Y")
    if corners[1][1] >= corners[2][1]: logger.warning(f"Top right Y is greater than bottom right Y")
    area = cv2.contourArea(corners)
    return area


def adjust_by_pov(Hs, corners):
    best_pov = None
    best_score = 0

    logger.info('Adjusting PoV ...')
    for refH in Hs:
        areas = []
        pov = np.linalg.inv(refH)
        for H in Hs:
            warp_corners = cv2.perspectiveTransform(corners, np.dot(pov, H))
            area = verify_corners(warp_corners)
            if area < 0:
                areas = []
                break
            areas.append(area)
        if not areas:
            continue
        score = min(areas) / max(areas)
        if best_pov is None or score > best_score:
            best_score = score
            best_pov = pov

    logger.info(f'==> Best PoV score: {best_score:.3f}')
    if (best_score <= 0.01):
        logger.warning(f'ERROR: Low best pov score!')
        # sys.exit('ERROR: Low best pov score!')

    if best_pov is None:
        return [H.copy() for H in Hs]
    else:
        return [np.dot(best_pov, H) for H in Hs]



def adjust_roi(homographies, corners, max_size):

    ## Get offset and size
    sl, st, sw, sh = get_boundingBox(homographies, corners)
    logger.info(f'Adjusting RoI of panorama ...{sl, st, sh, sw}')

    ## Rescale
    s = min(max_size / max(sw, sh), 1)
    offset = np.float32([s, 0, -s * sl, 0, s, -s * st, 0, 0, 1]).reshape(3, 3)
    homographies = [np.dot(offset, H) for H in homographies]
    dl, dt, dw, dh = get_boundingBox(homographies, corners)

    logger.info(f'==> Rescaled from {sw}x{sh} to {dw}x{dh}')
    logger.info(f'==> Offset from ({sl}, {st}) to ({dl}, {dt})')

    return homographies, dl, dt, dw, dh


def calculate_homography(imgs):
    Homographies = []
    num_imgs = len(imgs)
    for i in range(num_imgs-1):
        j = i + 1
        img0 = imgs[i]
        img1 = imgs[j]
        
        t_img0 = numpy_image_to_torch(img0).cuda()
        t_img1 = numpy_image_to_torch(img1).cuda()
            
            # extract local features
        feats0 = extractor_aliked.extract(t_img0)  # auto-resize the image, disable with resize=None
        feats1 = extractor_aliked.extract(t_img1)

        # match the features
        matches01 = matcher_aliked({'image0': feats0, 'image1': feats1})
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
        Homographies.append(M)
    
    Hs = [np.eye(3, 3, dtype=np.float32)]
    for i in range(len(Homographies)):
        H = np.dot(Hs[-1], np.float32(Homographies[i]).reshape(3, 3))
        Hs.append(H)
    return Hs


def rectify_horizontally(homographies, corners):
    ## Get edges
    left_pts, right_pts = [], []
    for H in homographies:
        warp_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        left_pts.extend([warp_corners[0], warp_corners[3]])
        right_pts.extend([warp_corners[1], warp_corners[2]])
    left_pts.sort(key=lambda v: v[0])
    right_pts.sort(key=lambda v: v[0])


    ## Get size
    l, t, w, h = get_boundingBox(homographies, corners)


    ## Get left edges: x = ay + b
    num = len(left_pts)
    left_edges = []
    for i1 in range(num):
        x1, y1 = left_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = left_pts[i2]
            if y1 == y2:
                continue
            a = (x1 - x2) / (y1 - y2)
            b = x1 - a * y1
            valid = True
            for x, y in left_pts + right_pts:
                if round(x) < round(y * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            xt = a * t + b
            xb = a * (t + h) + b
            left_edges.append((xt, xb))


    ## Get bottom edges: x = ay + b
    num = len(right_pts)
    right_edges = []
    for i1 in range(num):
        x1, y1 = right_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = right_pts[i2]
            if y1 == y2:
                continue
            a = (x1 - x2) / (y1 - y2)
            b = x1 - a * y1
            valid = True
            for x, y in left_pts + right_pts:
                if round(x) > round(y * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            xt = a * t + b
            xb = a * (t + h) + b
            right_edges.append((xt, xb))


    ## Find smallest quadrangle
    r,  b = l + w, t + h
    polys = []
    for tl, bl in left_edges:
        for tr, br in right_edges:
            if tl >= tr or bl >= br:
                continue
            poly = np.float32([tl, t, tr, t, br, b, bl, b]).reshape(4, 1, 2)
            area = cv2.contourArea(poly)
            polys.append((area, (tl, bl, tr, br)))
    if not polys:
        return homographies
    polys.sort(key=lambda x: x[0])
    area, poly = polys[0]
    if area > w * h:
        return homographies


    ## Align both edges
    tl, bl, tr, br = poly
    l, r = 0.5 * (tl + bl), 0.5 * (tr + br)
    src_pts = np.float32([tl, t, tr, t, br, b, bl, b]).reshape(4, 2)
    dst_pts = np.float32([l, t, r, t, r, b, l, b]).reshape(4, 2)
    T = cv2.getPerspectiveTransform(src_pts, dst_pts)
    homographies = [np.dot(T, H) for H in homographies]
    return homographies

def rectify_vertically(homographies, corners):


    ## Get edges
    top_pts, bottom_pts = [], []
    for H in homographies:
        warp_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        top_pts.extend([warp_corners[0], warp_corners[1]])
        bottom_pts.extend([warp_corners[3], warp_corners[2]])
    top_pts.sort(key=lambda v: v[1])
    bottom_pts.sort(key=lambda v: v[1])


    ## Get size
    l, t, w, h = get_boundingBox(homographies, corners)


    ## Get top edges: y = ax + b
    top_edges = []
    num = len(top_pts)
    for i1 in range(num):
        x1, y1 = top_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = top_pts[i2]
            if x1 == x2:
                continue
            a = (y1 - y2) / (x1 - x2)
            b = y1 - a * x1
            valid = True
            for x, y in top_pts + bottom_pts:
                if round(y) < round(x * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            yl = a * l + b
            yr = a * (l + w) + b
            top_edges.append((yl, yr))


    ## Get bottom edges: y = ax + b
    num = len(bottom_pts)
    bottom_edges = []
    for i1 in range(num):
        x1, y1 = bottom_pts[i1]
        for i2 in range(i1 + 1, num):
            x2, y2 = bottom_pts[i2]
            if x1 == x2:
                continue
            a = (y1 - y2) / (x1 - x2)
            b = y1 - a * x1
            valid = True
            for x, y in top_pts + bottom_pts:
                if round(y) > round(x * a + b):
                    valid = False
                    break
            if not valid:
                continue
            ## Find area
            yl = a * l + b
            yr = a * (l + w) + b
            bottom_edges.append((yl, yr))


    ## Find smallest quadrangle
    r,  b = l + w, t + h
    polys = []
    for tl, tr in top_edges:
        for bl, br in bottom_edges:
            if tl >= bl or tr >= br:
                continue
            poly = np.float32([l, tl, r, tr, r, br, l, bl]).reshape(4, 1, 2)
            area = cv2.contourArea(poly)
            polys.append((area, (tl, tr, bl, br)))
    if not polys:
        return homographies
    polys.sort(key=lambda x: x[0])
    area, poly = polys[0]
    if area > w * h:
        return homographies


    ## Align both edges
    tl, tr, bl, br = poly
    t, b = 0.5 * (tl + tr), 0.5 * (bl + br)
    src_pts = np.float32([l, tl, r, tr, r, br, l, bl]).reshape(4, 2)
    dst_pts = np.float32([l, t, r, t, r, b, l, b]).reshape(4, 2)
    T = cv2.getPerspectiveTransform(src_pts, dst_pts)
    homographies = [np.dot(T, H) for H in homographies]
    return homographies



def sort_key_func(item):  
    base_name = os.path.basename(item)  # get file name with extension  
    num = os.path.splitext(base_name)[0]  # remove extension  
    return int(num)  

def read_local_imgs(img_path):
    imgs = []
    img_files = glob.glob(img_path + "/*.jpg")
    img_files = sorted(img_files, key=sort_key_func)

    sep = 3
    img_files_new = []
    for i in range(len(img_files)):
        if i % sep == 0:
            img_files_new.append(img_files[i])
    # img_files = img_files[:2]
    print("img_files_new: ", len(img_files_new))
    for img_file in img_files_new:
        img = cv2.imread(img_file)
        imgs.append(img)
    return imgs


def stitch_local(img_folder):
    imgs= read_local_imgs(img_folder)
    print("imgs: ", len(imgs))
    img_size = imgs[0].shape[1], imgs[0].shape[0]
    # print("Hs: ", Hs)
    t0 = time.time()
    new_Hs = calculate_homography(imgs)
    t1 = time.time()
    print("time: ", t1-t0)
    # print("new_Hs: ", new_Hs)
    w, h = img_size
    print("w,h: ", w,h)
    corners = np.float32([0, 0, w, 0, w, h, 0, h]).reshape(4, 1, 2)
    
    Hs = new_Hs.copy()
    # Rescale homography with original image size
    # s = 360 / max(h, w)
    # offset = np.float32([s, 0, 0, 0, s, 0, 0, 0, 1]).reshape(3, 3)
    # Hs = [np.dot(H, offset) for H in Hs]
    
    Hs = adjust_by_pov(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    Hs = rectify_horizontally(Hs, corners)
    Hs = rectify_vertically(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    
    ## Generate panorama
    print("l, t, pw, ph", pw, ph)
    # assert abs(l) < 2 and abs(t) < 2, f'BUG #1 in stitch(): {l}, {t}'
    pano = np.zeros((ph, pw, 3), np.uint8)

    for img, H in zip(imgs, Hs):
        cv2.warpPerspective(img, H, (pw, ph), pano, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imwrite("stitch.jpg", pano)


def stitch(req_path):
    req = json.load(open(req_path))
    imgs, Hs, unsorted_names, img_size = parse_req(req)
    # print("Hs: ", Hs)
    t0 = time.time()
    new_Hs = calculate_homography(imgs)
    t1 = time.time()
    print("time: ", t1-t0)
    # print("new_Hs: ", new_Hs)
    w, h = img_size
    print("w,h: ", w,h)
    corners = np.float32([0, 0, w, 0, w, h, 0, h]).reshape(4, 1, 2)
    
    Hs = new_Hs.copy()
    # Rescale homography with original image size
    # s = 360 / max(h, w)
    # offset = np.float32([s, 0, 0, 0, s, 0, 0, 0, 1]).reshape(3, 3)
    # Hs = [np.dot(H, offset) for H in Hs]
    
    Hs = adjust_by_pov(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    Hs = rectify_horizontally(Hs, corners)
    Hs = rectify_vertically(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    
    ## Generate panorama
    print("l, t, pw, ph", pw, ph)
    # assert abs(l) < 2 and abs(t) < 2, f'BUG #1 in stitch(): {l}, {t}'
    pano = np.zeros((ph, pw, 3), np.uint8)

    for img, H in zip(imgs, Hs):
        cv2.warpPerspective(img, H, (pw, ph), pano, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imwrite("stitch.jpg", pano)

if __name__ == "__main__":
    # req_path = "/datadrive/codes/opensource/features/LightGlue/assets/uspg_test_jsons/4c89ccd3-5978-4d74-8764-7daf9d35cdda_input.json"
    # stitch(req_path)
    
    stitch_local("/datadrive/codes/retail/ultralytics/stitch/output/imgs")
    