#!/usr/bin/env python3

# https://stackoverflow.com/a/45639406
import sys

import cv2
import numpy as np

H_templ_ratio = 0.45  # H_templ_ratio: horizontal ratio of the input that we will keep to create a template


def genTemplate(img):
    # we get the image's width and height
    h, w = img.shape[:2]
    # we compute the template's bounds
    x1 = int(float(w)*(1-H_templ_ratio))
    y1 = 0
    x2 = w
    y2 = h
    return(img[y1:y2, x1:x2])  # and crop the input image


def mat2Edges(img):  # applies a Canny filter to get the edges
    return img
    edged = cv2.Canny(img, 100, 200)
    return(edged)


def addBlackMargins(img, top, bottom, left, right):  # top, bottom, left, right: margins width in pixels
    h, w = img.shape[:2]
    result = np.zeros((h+top+bottom, w+left+right, 3), np.uint8)
    result[top:top+h, left:left+w] = img
    return(result)

# return the y_offset of the first image to stitch and the final image size needed
def calcFinalImgSize(imgs, loc):
    y_offset = 0
    max_margin_top = 0
    max_margin_bottom = 0  # maximum margins that will be needed above and bellow the first image in order to stitch all the images into one mat
    current_margin_top = 0
    current_margin_bottom = 0

    h_init, w_init = imgs[0].shape[:2]
    w_final = w_init

    for i in range(0, len(loc)):
        h, w = imgs[i].shape[:2]
        h2, w2 = imgs[i+1].shape[:2]
        # we compute the max top/bottom margins that will be needed (relatively to the first input image) in order to stitch all the images
        # here, we assume that the template top-left corner Y-coordinate is 0 (relatively to its original image)
        current_margin_top += loc[i][1]
        current_margin_bottom += (h2 - loc[i][1]) - h
        if(current_margin_top > max_margin_top):
            max_margin_top = current_margin_top
        if(current_margin_bottom > max_margin_bottom):
            max_margin_bottom = current_margin_bottom
        # we compute the width needed for the final result
        # x-coordinate of the template relatively to its original image
        x_templ = int(float(w)*H_templ_ratio)
        w_final += (w2 - x_templ - loc[i][0])  # width needed to stitch all the images into one mat

    h_final = h_init + max_margin_top + max_margin_bottom
    return (max_margin_top, h_final, w_final)

# match each input image with its following image (1->2, 2->3)
def matchImages(imgs, templates_loc):
    for i in range(0, len(imgs)-1):
        template = genTemplate(imgs[i])
        template = mat2Edges(template)
        h_templ, w_templ = template.shape[:2]
        # Apply template Matching
        margin_top = margin_bottom = h_templ
        margin_left = margin_right = 0
        # we need to enlarge the input image prior to call matchTemplate (template needs to be strictly smaller than the input image)
        img = addBlackMargins(imgs[i+1], margin_top, margin_bottom, margin_left, margin_right)
        img = mat2Edges(img)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)  # matching function
        _, _, _, templ_pos = cv2.minMaxLoc(res)  # minMaxLoc gets the best match position
        # as we added margins to the input image we need to subtract the margins width to get the template position relatively to the initial input image (without the black margins)
        rectified_templ_pos = (templ_pos[0]-margin_left, templ_pos[1]-margin_top)
        templates_loc.append(rectified_templ_pos)
        print("max_loc", rectified_templ_pos)


def stitchImages(imgs, templates_loc):
    # we calculate the "surface" needed to stitch all the images into one mat (and y_offset, the Y offset of the first image to be stitched)
    y_offset, h_final, w_final = calcFinalImgSize(imgs, templates_loc)
    result = np.zeros((h_final, w_final, 3), np.uint8)

    # initial stitch
    h_init, w_init = imgs[0].shape[:2]
    result[y_offset:y_offset+h_init, 0:w_init] = imgs[0]
    origin = (y_offset, 0)  # top-left corner of the last stitched image (y,x)
    # stitching loop
    for j in range(0, len(templates_loc)):
        h, w = imgs[j].shape[:2]
        h2, w2 = imgs[j+1].shape[:2]
        # we compute the coordinates where to stitch imgs[j+1]
        y1 = origin[0] - templates_loc[j][1]
        y2 = origin[0] - templates_loc[j][1] + h2
        # x-coordinate of the template relatively to its original image's right side
        x_templ = int(float(w)*(1-H_templ_ratio))
        x1 = origin[1] + x_templ - templates_loc[j][0]
        x2 = origin[1] + x_templ - templates_loc[j][0] + w2
        result[y1:y2, x1:x2] = imgs[j+1]  # we copy the input image into the result mat
        origin = (y1, x1)  # we update the origin point with the last stitched image

    return(result)


if __name__ == '__main__':
    paths = sys.argv[1:]
    imgs = [cv2.imread(path) for path in paths if not path.startswith('--')]

    templates_loc = []  # templates location
    matchImages(imgs, templates_loc)

    result = stitchImages(imgs, templates_loc)
    cv2.imshow("result", result)
    cv2.waitKey(0)
