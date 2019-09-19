#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF signal processing example - pruning

@author: duc_hoang
"""

#Set up
# Import all the things we need ---
import tensorflow as tf
import tensorboard
import tensorflow.keras as keras

import numpy as np
import _pickle as cPickle

import tempfile
logdir = tempfile.mkdtemp()

import os
os.makedirs('model',exist_ok=True)

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

#Build the models
print("Preparing the model ...")
print("Input shape ", in_shp)
print("Using Keras version: ", keras.__version__)

#Larger network with convolutional layers
l = tf.keras.layers

dr = 0.5 # dropout rate (%)

model = tf.keras.Sequential([
        l.Reshape([1]+in_shp, input_shape=in_shp),
        l.Conv2D(64, (2, 8), padding='valid', data_format="channels_first", activation="relu", name="conv1", kernel_initializer='glorot_uniform'),
        l.Dropout(dr),
        l.Conv2D(32, (1, 32), padding='valid', data_format="channels_first", activation="relu", name="conv2", kernel_initializer='glorot_uniform'),
        l.Dropout(dr),
        l.Flatten(),
        l.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"),
        l.Dropout(dr),
        l.Dense(len(classes), kernel_initializer='he_normal', name="dense2"),
        l.Activation('softmax')
        ])
    
model.summary()



model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam')

#================TRAIN THE MODEL====================
# Set up some params 
nb_epoch = 50     # number of epochs to train on
batch_size = 1024  # training batch size

filepath = 'model/convmodrecnets_CNN2_0.5.wts.h5'

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0),
             tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
             ]

history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=(X_test, Y_test))

#print out accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#=====================PRUNING=======================
