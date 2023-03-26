#!/usr/bin/env python3

import cProfile
import sys
from collections import Counter
from functools import reduce

import cv2
import numpy as np


USE_SIFT = True
ALIGNMENT_SCORE_THRESHOLD = 0.015
TOP_BOTTOM_THRESHOLD = 10
GREEDY_HEADER = False
HORIZONTAL_MARGIN = 0
REFERENCE_HEADER_IMAGE = 0


def diff_images(img1, img2):
  h = min(img1.shape[0], img2.shape[0])
  w = min(img1.shape[1], img2.shape[1])

  diff = cv2.absdiff(img1[:h,:w], img2[:h,:w])
  mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  boolmask = mask < TOP_BOTTOM_THRESHOLD  # type: ignore

  row_threshold = w - 20
  row_sum = np.sum(boolmask, axis=1)
  row_mask = row_sum < row_threshold
  index_array = np.where(row_mask)[0]

  return index_array[0], index_array[-1]


def find_features(imgs):
  gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
  if USE_SIFT:
    sift = cv2.SIFT_create()
    return [sift.detectAndCompute(img, None) for img in gray_imgs]
  else:
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    return [brief.compute(img, star.detect(img, None)) for img in gray_imgs]


def crop_images(imgs):
  if REFERENCE_HEADER_IMAGE == -1:
    return imgs, imgs[0][:0,:0,:], imgs[0][:0,:0,:]

  j = min(REFERENCE_HEADER_IMAGE, len(imgs)-1)
  bounds = [diff_images(imgs[j], imgs[i]) for i in range(len(imgs)) if i != j]
  if GREEDY_HEADER:
    min_bound = max(bounds, key=lambda x: x[0])[0]
    max_bound = min(bounds, key=lambda x: x[1])[1]
  else:
    min_bound = min(bounds, key=lambda x: x[0])[0]
    max_bound = max(bounds, key=lambda x: x[1])[1]

  mslice = slice(HORIZONTAL_MARGIN, -HORIZONTAL_MARGIN)
  if HORIZONTAL_MARGIN == 0:
    mslice = slice(None, None)
  cropped_imgs = [img[min_bound:max_bound,mslice,:] for img in imgs]
  top = imgs[0][:min_bound,mslice,:]
  bottom = imgs[-1][max_bound:,mslice,:]
  return cropped_imgs, top, bottom


def match_features(feat1, feat2):
  kp1, des1 = feat1
  kp2, des2 = feat2

  # BFMatcher with default params
  bf = cv2.BFMatcher()
  try:
    matches = bf.knnMatch(des1, des2, k=2)
  except cv2.error as e:
    print(e)
    return []

  # Apply ratio test
  deltas = []
  for m, n in matches:
    if m.distance < 0.75*n.distance:
      if kp1[m.queryIdx].pt != kp2[m.trainIdx].pt:
        pt1, pt2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
        deltas.append((int(pt1[0] - pt2[0]), int(pt1[1] - pt2[1])))

  return [(dx,dy) for (dx,dy), c in Counter(deltas).most_common(10) if c > 1]


# https://stackoverflow.com/a/25068722
class Rectangle:
  def __and__(self, other: "Rectangle"):
    a, b = self, other
    x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
    y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
    x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
    y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
    return Rectangle(x1, y1, x2, y2)

  def __or__(self, other: "Rectangle"):
    a, b = self, other
    x1 = min(a.x1, a.x2, b.x1, b.x2)
    y1 = min(a.y1, a.y2, b.y1, b.y2)
    x2 = max(a.x1, a.x2, b.x1, b.x2)
    y2 = max(a.y1, a.y2, b.y1, b.y2)
    return Rectangle(x1, y1, x2, y2)

  def __init__(self, x1, y1, x2, y2):
    if x1>x2 or y1>y2:
      raise ValueError("Coordinates are invalid")
    self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

  def __add__(self, point):
    dx, dy = point
    x1 = self.x1 + dx
    x2 = self.x2 + dx
    y1 = self.y1 + dy
    y2 = self.y2 + dy
    return Rectangle(x1, y1, x2, y2)

  def __sub__(self, point):
    dx, dy = point
    x1 = self.x1 - dx
    x2 = self.x2 - dx
    y1 = self.y1 - dy
    y2 = self.y2 - dy
    return Rectangle(x1, y1, x2, y2)

  def __repr__(self):
    return f"Rectangle({self.x1}, {self.y1}, {self.x2}, {self.y2})"

  @staticmethod
  def from_xywh(x, y, w, h):
    x1, y1, x2, y2 = x, y, x+w, y+h
    return Rectangle(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

  @staticmethod
  def from_img(img):
    return Rectangle.from_xywh(0, 0, img.shape[0], img.shape[1])

  @property
  def w(self):
    return self.x2 - self.x1

  @property
  def h(self):
    return self.y2 - self.y1


def show_boolmask(mask, block):
  canvas = np.zeros_like(block, np.uint8)
  canvas[:,:,0] = 0xff
  canvas[:,:,2] = 0xff
  canvas[mask] = block[mask]
  cv2.imshow("canvas", canvas)


def overlay_score(img1, img2, delta: tuple[int, int]):
  r1 = Rectangle.from_img(img1)
  r2 = Rectangle.from_img(img2) + delta
  r3 = r1 & r2
  r4 = r3 - delta

  block1 = img1[r3.x1:r3.x2,r3.y1:r3.y2,:]
  block2 = img2[r4.x1:r4.x2,r4.y1:r4.y2,:]
  diff = cv2.absdiff(block1, block2)
  mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  boolmask = mask < 30  # type: ignore
  # show_boolmask(boolmask, block1)
  return np.sum(~boolmask) / boolmask.size


def overlay_images(imgs, deltas: list[tuple[int, int]]):
  rects = [Rectangle.from_img(img) + delta for img, delta in zip(imgs, deltas)]
  r3 = reduce(lambda x,y: x|y, rects, rects[0])
  rects2 = [rect - (r3.x1, r3.y1) for rect in rects]

  res = np.zeros((r3.w,r3.h,3), dtype='uint8')
  for rect, img in zip(rects2, imgs):
    res[rect.x1:rect.x2,rect.y1:rect.y2,:] = img
  return res


def find_top_bottom(imgs, deltas: list[tuple[int, int]]):
  rects = [Rectangle.from_img(img) + delta for img, delta in zip(imgs, deltas)]
  r3 = reduce(lambda x,y: x|y, rects, rects[0])
  return r3.x1, r3.x2


def main(paths, outpath):
  imgs = [cv2.imread(path) for path in paths if not path.startswith('--')]
  cropped_imgs, top, bottom = crop_images(imgs)
  features = find_features(cropped_imgs)

  overlay_deltas: list[tuple[int, int]] = [(0,0)]*len(imgs)

  for j in range(1, len(features)):
    for i in reversed(range(0, j)):
      deltas = match_features(features[i], features[j])
      scores = [overlay_score(cropped_imgs[i], cropped_imgs[j], delta[::-1]) for delta in deltas]

      if min(scores, default=1) > ALIGNMENT_SCORE_THRESHOLD:
        print()
        for score, delta in zip(scores, deltas):
          print(f"img{i} and img{j} with delta {delta} = {score}")
        print(f"img{i} and img{j} DO NOT MATCH!!")
        print()
        continue

      k = np.argmin(scores)
      print(f"mathed img{i} and img{j} with delta {deltas[k]} = {scores[k]}")
      x,y = overlay_deltas[i]
      dx,dy = deltas[k][::-1]
      overlay_deltas[j] = (x+dx, y+dy)
      break
    else:
      print(f"img{j} does not know where to go :(")
      _, bottomx = find_top_bottom(cropped_imgs, overlay_deltas)
      _,y = overlay_deltas[j-1]
      overlay_deltas[j] = (bottomx+10, y)

  topx, bottomx = find_top_bottom(cropped_imgs, overlay_deltas)
  overlay_deltas.insert(0, (topx-top.shape[0], 0))
  overlay_deltas.append((bottomx, overlay_deltas[-1][1]))
  print(overlay_deltas)
  res = overlay_images([top] + cropped_imgs + [bottom], overlay_deltas)
  cv2.imwrite(outpath, res)


output_path = sys.argv[-1]
image_paths = sys.argv[1:-1]
main(image_paths, output_path)
# cProfile.run("main(image_paths, output_path)")
