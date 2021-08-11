#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 01:56:42 2021

@author: a975193
"""

import cv2
import glob
import numpy as np

input_path = "../sample_data/"
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("display", cv2.WINDOW_NORMAL)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3 ))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 600:
        im_scale = float(600) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im#, (new_h / img_size[0], new_w / img_size[1])

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=3)
    return thresh

def get_MRZ_contour(image, frame):
    contours, hier = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("contours", frame)

for image_path in glob.glob(input_path+"*.jpg"):
        frame = cv2.imread(image_path)
        frame = resize_image(frame)
        cv2.imshow("frame", frame)
        #frame = cv2.resize(frame, (480,640), cv2.INTER_CUBIC)
        print(frame.shape)
        threshold_image = process_image(frame)
        
        cv2.imshow("display", threshold_image)
        get_MRZ_contour(threshold_image, frame)
        cv2.waitKey(0)
cv2.destroyAllWindows()
