#!/usr/bin/env python3

import sys
import cv2
import imutils

# https://forum.opencv.org/t/how-to-stitch-two-screenshots-with-open-cv-and-surf/3337

rotate = '--rotate' in sys.argv
image_paths = sys.argv[1:-1]
imgs = [cv2.imread(path) for path in image_paths if not path.startswith('--')]

for i, img in enumerate(imgs):
  cv2.imshow(str(i+1), img)

if rotate:
  for i, img in enumerate(imgs):
    imgs[i] = imutils.rotate_bound(img, 90)

stitcher = cv2.Stitcher.create()
error_code, output = stitcher.stitch(imgs)

if error_code != cv2.STITCHER_OK:
  print("stitching ain't successful")
else:
  print('Your Panorama is ready!!!')

# final output
if rotate:
  output = imutils.rotate_bound(output, -90)
cv2.imshow('final result', output)
cv2.imwrite(sys.argv[-1], output)

cv2.waitKey(0)
