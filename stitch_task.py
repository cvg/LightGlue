from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
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
from lightglue import LightGlue, SIFT, ALIKED
from lightglue.utils import numpy_image_to_torch, rbd

# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher

# SIFT+LightGlue
extractor_sift = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_sift = LightGlue(features='sift').eval().cuda()  # load the matcher


logger = loguru.logger
logger.add("stitch_task.log", format="{time} {level} {message}", level="INFO", rotation="1 MB", compression="zip")

DOWNLOAD_CACHE_DIR = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/imgs"
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

    if path and path.is_file():
        return parse_func(path.read_bytes())

    ## Download
    for _ in range(num_retry):
        try:
            rsp = requests.get(url, timeout=60)
            if rsp.status_code == 200:
                result = parse_func(rsp.content)
                if path: path.write_bytes(rsp.content)
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
        assert num_imgs == len(homographies), f'Invalid homography number {len(homographies)}, expected {num_imgs}'
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
        assert num_imgs == 1 + len(req['pair']), f"Invalid homography number {len(req['pair'])}, expected {num_imgs - 1}"
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
    logger.info(f'Adjusting RoI of panorama ...')

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
        h, w = img1.shape[:2]
        
        t_img0 = numpy_image_to_torch(img0).cuda()
        t_img1 = numpy_image_to_torch(img1).cuda()
            
            # extract local features
        feats0 = extractor_sift.extract(t_img0)  # auto-resize the image, disable with resize=None
        feats1 = extractor_sift.extract(t_img1)

        # match the features
        matches01 = matcher_sift({'image0': feats0, 'image1': feats1})
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


def stitch(req_path):
    req = json.load(open(req_path))
    imgs, Hs, unsorted_names, img_size = parse_req(req)
    # print("Hs: ", Hs)
    new_Hs = calculate_homography(imgs)
    # print("new_Hs: ", new_Hs)
    w, h = img_size
    print("w,h: ", w,h)
    corners = np.float32([0, 0, w, 0, w, h, 0, h]).reshape(4, 1, 2)
    
    Hs = new_Hs.copy()
    ## Rescale homography with original image size
    # s = 360 / max(h, w)
    # offset = np.float32([s, 0, 0, 0, s, 0, 0, 0, 1]).reshape(3, 3)
    # Hs = [np.dot(H, offset) for H in Hs]
    
    Hs = adjust_by_pov(Hs, corners)
    Hs, l, t, pw, ph = adjust_roi(Hs, corners, 10000)
    
    ## Generate panorama
    print("l, t, pw, ph", pw, ph)
    assert abs(l) < 2 and abs(t) < 2, f'BUG #1 in stitch(): {l}, {t}'
    pano = np.zeros((ph, pw, 3), np.uint8)

    for img, H in zip(imgs, Hs):
        cv2.warpPerspective(img, H, (pw, ph), pano, borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imwrite("stitch.jpg", pano)

if __name__ == "__main__":
    # req_normal_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/ae52a0b3-4fc2-4cf8-8c5c-61befb8feb54_input.json"
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/c73828ec-d806-433f-90c1-6a2cc28ad80d_input.json" # BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/57370a01-61f3-4614-b079-8d210551dc4f_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/ec9866a6-7186-444f-b856-db78ebac2130_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/dd429d25-8cde-4562-8a22-9887c524809d_input.json" #BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/42c6c651-36e1-4b64-9a74-3a654a9a204a_input.json" #BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/0d5ea595-91bc-4906-a325-b25b75d8ddad_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/779521df-3617-49dd-8c88-c88381bea6e2_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/a9b1a4af-b735-4966-b642-0733c0afa617_input.json" #BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/888d3745-6951-48fa-9689-87013ab246f9_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/5779be24-2079-4706-a881-8cc17a8a4346_input.json" #BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/d3bed88c-65dd-46ba-8506-f14279ebe8d4_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/d1404239-d77e-4b3e-ba9c-dba4e4b954f0_input.json" #BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/8c4783df-fc59-48d0-a394-a9cb27fb7a46_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/1b92ad6b-9ad4-4430-9e07-376fa6aad7bb_input.json" #OK
    req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/2d3c5d1a-0b08-4908-a9d9-0215be9ec7d1_input.json" #OK
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/dd1798ca-ae6c-4afe-bbb4-97a140a555fe_input.json" #BAD
    # req_path = "/datadrive/codes/opensource/dlfeatures/LightGlue/assets/new/8c8d8417-6cb6-484c-8631-54df1dbcabf6_input.json" #BAD
    
    stitch(req_path)
    