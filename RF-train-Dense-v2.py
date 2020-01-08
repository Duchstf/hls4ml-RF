#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:14:03 2019
@author: duc_hoang
"""

# Import all the things we need ---
# by setting env variables before Keras import you can set up which backend and which GPU it uses
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D
import _pickle as cPickle
import keras

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

in_shp = list(X_train.shape[1:])
print("X_train shape: ", X_train.shape)
classes = mods



#================BUILD THE MODEL====================

print("Preparing the model ...")
print("Input shape ", in_shp)
print("Using Keras version: ", keras.__version__)


# Dense-only network
dr = 0.5 # dropout rate (%)

model = models.Sequential()
model.add(Reshape([1, in_shp[0]*in_shp[1]], input_shape=in_shp, name = "input_1"))
model.add(Dense(500, activation="relu", kernel_initializer="he_normal", name = "fc_relu1"))
model.add(Dropout(rate=dr, name = "dropout_1"))
model.add(Dense(500, activation="relu", kernel_initializer="he_normal", name = "fc_relu2"))
model.add(Dropout(rate=dr, name = "dropout_2"))
model.add(Dense(500, activation="relu", kernel_initializer="he_normal", name = "fc_relu3"))
model.add(Dropout(rate=dr, name = "dropout_3"))
model.add(Dense(len(classes), activation="relu", kernel_initializer="he_normal", name = "fc_relu4"))
model.add(Activation('softmax', name = "softmax_activation4"))
model.add(Reshape([len(classes)], name = "output"))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Save the model architecture
with open('./model/RF-Dense-full.json', 'w') as f:
    f.write(model.to_json())


    
#================TRAIN THE MODEL====================

# Set up some params 
nb_epoch = 50     # number of epochs to train on
batch_size = 1024  # training batch size

filepath = 'model/RF-Dense-full.h5'

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