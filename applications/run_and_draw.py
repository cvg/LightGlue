
import numpy as np
import cv2
import json
import time
import sys

sys.path.append("/datadrive/codes/opensource/features/LightGlue")

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue.viz2d import plot_images, plot_keypoints, plot_matches, save_plot


# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
img0_path = "/datadrive/codes/opensource/features/LightGlue/assets/CAPG/2.jpg"
img1_path = "/datadrive/codes/opensource/features/LightGlue/assets/CAPG/3.jpg"
# size = (640, 480)
image0 = load_image(img0_path).cuda()
image1 = load_image(img1_path).cuda()
print("image0: ", image0.shape)
print("image1: ", image1.shape)

t0 = time.time()
# extract local features
feats0 = extractor_aliked.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor_aliked.extract(image1)

# match the features
matches01 = matcher_aliked({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

t1 = time.time()
print("time: ", t1-t0)

# plot and save the matches
points0 = points0.cpu().numpy()
points1 = points1.cpu().numpy()
plot_images([image0, image1])
print("points0: ", points0.shape, points0)
# plot_keypoints(points0)
# plot_keypoints(points1)
plot_matches(points0, points1, color='lime', lw=1)
save_plot("matches.png")


# anslysis
x0 = points0[:, 0]
y0 = points0[:, 1]
x1 = points1[:, 0]
y1 = points1[:, 1]

x_10 = x1 - x0
y_10 = y1 - y0
print("x_10: ", np.mean(x_10), np.std(x_10), np.median(x_10))
print("y_10: ", np.mean(y_10), np.std(y_10), np.median(y_10))

