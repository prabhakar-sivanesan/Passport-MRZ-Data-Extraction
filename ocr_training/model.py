#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:12:27 2021

@author: prabhakar
"""
# Build basic Multilayer perceptron

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop

def base_model(classes, input_shape):

    model = Sequential()
    model.add(Dense(512, activation = "relu", input_shape = (21,21,1)))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(len(classes), activation = "softmax"))
    
    model.compile(optimizer=RMSprop(learning_rate = 0.005), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model