#!/usr/bin/env python3

import sys
import cv2

margin = 100
image_path = sys.argv[1]
output_format_string = sys.argv[2]

image = cv2.imread(image_path)
w, h, _ = image.shape

res = []
for i in range(3):
  for j in range(3):
    res.append(image[w//3*i:w//3*(i+1) + margin, h//3*j:h//3*(j+1) + margin])

for i, img in enumerate(res):
  cv2.imwrite(output_format_string % i, img)

stitcher = cv2.Stitcher.create()
error_code, output = stitcher.stitch(res)

if error_code != cv2.STITCHER_OK:
  print("stitching ain't successful")
else:
  print('Your Panorama is ready!!!')

# final output
cv2.imshow('final result', output)
cv2.waitKey(0)
