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
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import _pickle as cPickle
import random, sys, keras

#==================PREPARE DATA=====================

# Prepare the training data
# You will need to seperately download or generate this file
Xd = cPickle.load(open("../RML2016.10a_dict.pkl",'rb'), encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
# into training and test sets of the form we can train/test on 
# while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

#Flip the last two dimensions
X_train = X_train.reshape((X_train.shape[0],X_train.shape[2],X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[2],X_test.shape[1]))

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

in_shape = list(X_train.shape[1:])
print("X_train shape: ", X_train.shape)
classes = mods
num_classes = len(classes)
print("Number of classes: ", num_classes)

#================BUILD THE MODEL====================

print("Preparing the model ...")
print("Input shape ", in_shape)
print("Using Keras version: ", keras.__version__)

# Conv1d model
dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Conv1D(200, 10, padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform', input_shape=in_shape))
model.add(Dropout(dr))
model.add(Conv1D(100, 10, padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics = ["accuracy"])
model.summary()

# Save the model architecture
with open('./model/v2-RF-CNN-full.json', 'w') as f:
    f.write(model.to_json())

#================TRAIN THE MODEL====================

# Set up some params 
nb_epoch = 50     # number of epochs to train on
batch_size = 1024  # training batch size

filepath = 'model/v2-RF-CNN-full.h5'

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
