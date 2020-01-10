#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:33:49 2019

@author: duc_hoang
"""

#===========================IMPORT STUFFS================================
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

%matplotlib inline
import _pickle as cPickle
from resnets_utils import *
from resnet_implementation import ResNet18

from keras import applications
from keras.layers import Dense, Dropout

#===========================PREPARE DATA================================

# Load the dataset ...
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

#Change the last two dimension of input to a square one
X_train = X_train.reshape((X_train.shape[0],16,16))
X_test = X_test.reshape((X_test.shape[0],16,16))

#Add empty dimension at the end for CONV input layer
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#Take in input shape and classes
input_shape = list(X_train.shape[1:])
classes = mods
num_classes = len(classes)

#===========================BUILD THE RESNET18 MODEL================================

base_model = ResNet18(in_shp, num_classes)
x = base_model.output
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

#===========================SAVE MODEL AND TRAIN================================

# Save the model architecture
with open('./model/RF-ResNet18.json', 'w') as f:
    f.write(model.to_json())

# Set up some params 
nb_epoch = 50     # number of epochs to train on
batch_size = 1024  # training batch size

filepath = 'model/RF-ResNet18.h5'

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    ])

