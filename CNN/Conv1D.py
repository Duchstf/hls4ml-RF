#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:33:49 2019
@author: duc_hoang
"""

# Import all the things we need ---
# by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling1D, ZeroPadding2D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import _pickle as cPickle
import random, sys, keras

#==================PREPARE DATA=====================

# Prepare the training data
# You will need to seperately download or generate this file
Xd = cPickle.load(open("RML2016.10a_dict.pkl",'rb'), encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data into training, validation, and test sets while keeping SNR and Mod labels handy for each
np.random.seed(2016)

#Number of samples
n_examples = X.shape[0]
n_train = int(n_examples * 0.6)
n_val = int(n_examples * 0.2)

train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
val_idx = np.random.choice(list(set(range(0,n_examples))-set(train_idx)), size=n_val, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))

X_train = X[train_idx]
X_val = X[val_idx]
X_test =  X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

#Flip the last two dimensions
X_train = X_train.reshape((X_train.shape[0],X_train.shape[2],X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0],X_val.shape[2],X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[2],X_test.shape[1]))

#Print out the shapes for different datasets
print ("Total number of samples: ", X.shape[0])
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of validation examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

print ("X_val shape: " + str(X_val.shape))
print ("Y_val shape: " + str(Y_val.shape))

print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#Input shape and number of classes
in_shape = list(X_train.shape[1:])
classes = mods
num_classes = len(classes)

print("X_train shape: ", X_train.shape)
print("Number of classes: ", num_classes)

#================BUILD THE MODEL====================

print("Preparing the model ...")
print("Input shape ", in_shape)
print("Using Keras version: ", keras.__version__)

# Conv1d model
model = models.Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape= in_shape))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Save the model architecture
with open('./model/Conv1D-full.json', 'w') as f:
    f.write(model.to_json())

#================TRAIN THE MODEL====================

# Set up some params 
nb_epoch = 50    # number of epochs to train on
batch_size = 1024  # training batch size

filepath = 'model/Conv1D-full.h5'

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
