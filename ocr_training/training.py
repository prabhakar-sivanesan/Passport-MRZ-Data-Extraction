#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 23:07:27 2021

@author: prabhakar
"""

import cv2
from model import base_model
from sklearn.datasets import load_files
from util.preprocessing import preprocess
from sklearn.model_selection import train_test_split

dataset_path = "../dataset/data/"

dataset = load_files(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(dataset.filenames, 
                                                    dataset.target, 
                                                    test_size = 0.3, shuffle=True)
classes = dataset.target_names

# read images
X_train = [cv2.imread(path, 0) for path in X_train]
X_test = [cv2.imread(path, 0) for path in X_test]

print("No of training samples: ", X_train[0].shape)
print("No of testing samples: ", X_test[0].shape)


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

model = base_model(classes, input_shape = (21,21,3))
print(model.summary())

train_generator, validation_generator = preprocess(X_train, X_test, y_train,
                                                   y_test, classes,
                                                   train_batch_size = 32,
                                                   test_batch_size = 20)

history = model.fit_generator(train_generator, validation_data= validation_generator, 
                              steps_per_epoch=1011, epochs = 20,
                              verbose=2, validation_steps=50)
model.save("trained_models/")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()