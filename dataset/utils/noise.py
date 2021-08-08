#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 21:19:05 2021

@author: prabhakar
"""

from skimage.util import random_noise

def add_salt_pepper(image):
    image = random_noise(image, mode = "s&p", clip=True)
    return (image*255.0).astype("uint8")

def add_speckle(image):
    image = random_noise(image, mode = "speckle", clip=True)
    return (image*255.0).astype("uint8")

