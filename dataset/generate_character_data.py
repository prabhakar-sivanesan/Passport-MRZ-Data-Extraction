#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 20:29:20 2021

@author: prabhakar
"""

import os
import cv2
import string
import numpy as np
from utils.draw_text import draw_text
from utils.noise import add_salt_pepper, add_speckle


b,g,r,a = 0,0,0,0
dataset_count = 100
blank_image = np.zeros((21,21,3),np.uint8)
blank_image.fill(255)

output_path = "data/"     
string_list = list(string.ascii_uppercase+"<")
fontpath = "assets/ocrb/OCRB Regular.ttf"

def main():
    
    for string_data in string_list:
        if not os.path.exists(output_path + string_data):
            os.mkdir(output_path+string_data)
        for i in range(dataset_count):
            img = draw_text(blank_image.copy(), fontpath, string_data)
            if i%2 == 0:
                img = add_salt_pepper(img)
            else:
                img = add_speckle(img)
            cv2.imshow("res", img)
            output_filepath = output_path+string_data+"/"+str(i+1)+".jpg"
            cv2.imwrite(output_filepath, img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()