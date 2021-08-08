#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 23:07:27 2021

@author: prabhakar
"""

import cv2
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

dataset_path = "../dataset/data/"

dataset = load_files(dataset_path)
classes = dataset.target_names
X_train, X_test, y_train, y_test = train_test_split(dataset.filenames, 
                                                    dataset.target, 
                                                    test_size = 0.3, shuffle=True)

print("No of training samples: ", len(X_train))
print("No of testing samples: ", len(X_test))
X_train = [cv2.imread(path) for path in X_train]
X_test = [cv2.imread(path) for path in X_test]

# visualize training and testing data
# Set up matplotlib fig, and size it to fit 4x4 pics
import matplotlib.pyplot as plt
import random
nrows = 4
ncols = 4

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

for i in range(nrows*ncols):
  n_train = random.randint(0,len(X_train))
  n_test = random.randint(0, len(X_test))

  sp = plt.subplot(nrows, ncols, i + 1)
  sp.set_title(classes[y_train[n_train]])
  plt.axis("off")
  plt.imshow(cv2.cvtColor(X_train[n_train], cv2.COLOR_BGR2RGB))

plt.show()
