#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 21:28:51 2021

@author: prabhakar
"""
# dataset preprocessing
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess(X_train, X_test, y_train, y_test, classes, 
               train_batch_size, test_batch_size):
    train_datagen = ImageDataGenerator( rescale= None )
    test_datagen = ImageDataGenerator( rescale= None )
    y_train = to_categorical(y_train, len(classes))
    y_test = to_categorical(y_test, len(classes))
    X_train = np.asarray([np.expand_dims(image, axis=2) for image in X_train]).astype("float32")/255
    X_test = np.asarray([np.expand_dims(image, axis=2) for image in X_test]).astype("float32")/255
    
    print("Training sample shape: ", X_train.shape)
    print("Testing sample shape: ", X_test.shape)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=train_batch_size, shuffle=True)
    validation_generator = test_datagen.flow(X_test, y_test, batch_size=test_batch_size, shuffle=True)
    return train_generator, validation_generator