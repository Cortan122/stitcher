#!/usr/bin/env python3

import sys
import cv2
import numpy as np

img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])

w = min(img1.shape[0], img2.shape[0])
h = min(img1.shape[1], img2.shape[1])

diff = cv2.absdiff(img1[:w,:h], img2[:w,:h])
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

th = 1
imask = mask<th

canvas = np.zeros_like(img2[:w,:h], np.uint8)
canvas[:,:,0] = 0xff
canvas[:,:,2] = 0xff
canvas[imask] = img2[:w,:h][imask]

cv2.imwrite("result.png", canvas)
