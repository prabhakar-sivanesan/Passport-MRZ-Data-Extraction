#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 21:45:35 2021

@author: prabhakar
"""

import numpy as np
from PIL import ImageFont, ImageDraw, Image

def draw_text(image, path, string_data):
    
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(path, 21)
    draw.text((3, 1),  string_data, font = font, fill = (255, 255, 255, 0))
    img = np.array(img_pil)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img