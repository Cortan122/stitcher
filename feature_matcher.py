#!/usr/bin/env python3

import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cProfile
from collections import Counter

def fast_detectAndCompute(img):
  # Initiate FAST detector
  star = cv.xfeatures2d.StarDetector_create()
  # Initiate BRIEF extractor
  brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
  # find the keypoints with STAR
  kp = star.detect(img, None)
  # compute the descriptors with BRIEF
  return brief.compute(img, kp)

def overlay_images(img1, img2, delta):
  h = max(img1.shape[0], img2.shape[0] + delta[0])
  w = max(img1.shape[1], img2.shape[1] + delta[1])
  res = np.zeros((h,w,3), dtype='uint8')
  res[:img1.shape[0],:img1.shape[1],:] = img1
  res[delta[0]:img2.shape[0] + delta[0],delta[1]:img2.shape[1] + delta[1],:] = img2
  return res

# https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
def main():
  img1 = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)  # queryImage
  img2 = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)  # trainImage

  # find the keypoints and descriptors with SIFT
  kp1, des1 = fast_detectAndCompute(img1)
  kp2, des2 = fast_detectAndCompute(img2)

  # BFMatcher with default params
  bf = cv.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2)

  # Apply ratio test
  deltas = []
  good = []
  for m, n in matches:
    if m.distance < 0.75*n.distance:
      if kp1[m.queryIdx].pt != kp2[m.trainIdx].pt:
        good.append([m])
        pt1, pt2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
        deltas.append((int(pt1[0] - pt2[0]), int(pt1[1] - pt2[1])))

  deltas = [(dx,dy) for (dx,dy), c in Counter(deltas).most_common(10) if c > 1]
  img4 = overlay_images(cv.imread(sys.argv[1]), cv.imread(sys.argv[2]), deltas[0][::-1])
  print(img4.shape, img1.shape, img2.shape, cv.imread(sys.argv[1]).shape)
  cv.imshow("overlay", img4)
  # cv.drawMatchesKnn expects list of lists as matches.
  return cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img3 = main()
plt.imshow(img3)
plt.show()

# cProfile.run('main()')
