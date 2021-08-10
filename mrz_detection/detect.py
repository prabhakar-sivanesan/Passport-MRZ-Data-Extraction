#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 01:56:42 2021

@author: a975193
"""

import cv2

input_path = "../sample_data/sample_1.jpg"
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("display", cv2.WINDOW_NORMAL)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)
    return thresh

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        
        cv2.imshow("frame", frame)
        frame = cv2.resize(frame, (640,480), cv2.INTER_CUBIC)
        print(frame.shape)
        image = process_image(frame)
        cv2.imshow("display", image)
        cv2.waitKey(1)
cv2.destroyAllWindows()
