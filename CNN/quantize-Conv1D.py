#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:33:49 2019
@author: duc_hoang
"""

import os,random, sys
import _pickle as cPickle
import numpy as np

import argparse

from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

#Use Qkeras for quantized layers
from qkeras import *

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

#Set up training parameters
classes = mods
IN_SHAPE = list(X_train.shape[1:])
NUM_CLASSES = len(classes)
OPTIMIZER = Adam()
NB_EPOCH = 50
BATCH_SIZE = 1024

print("X_train shape: ", X_train.shape)
print("Number of classes: ", NUM_CLASSES)

#================BUILD THE MODEL====================

print("Preparing the model ...")
print("Input shape ", in_shape)
print("Using Keras version: ", keras.__version__)

# Quantized Conv1D Model 
def QConv1D_model(weights_f, load_weights = False):
    """Construc QConv1D model"""
    x = x_in = Input(IN_SHAPE, name="input")
    X = QConv1D(filters=128, kernel_size=3, kernel_quantizer="stochastic_ternary", bias_quantizer="ternary", name="conv1d_1")(x)
    x = QActivation("quantized_relu(3)")(x)
    X = QConv1D(filters=128, kernel_size=3, kernel_quantizer="stochastic_ternary", bias_quantizer="ternary", name="conv1d_2")(x)
    x = QActivation("quantized_relu(3)")(x)
    x = MaxPooling1D(pool_size=2)(x)
    X = QConv1D(filters=64, kernel_size=3, kernel_quantizer="stochastic_ternary", bias_quantizer="ternary", name="conv1d_3")(x)
    x = QActivation("quantized_relu(3)")(x)
    X = QConv1D(filters=64, kernel_size=3, kernel_quantizer="stochastic_ternary", bias_quantizer="ternary", name="conv1d_4")(x)
    x = QActivation("quantized_relu(3)")(x)
    x = Dropout(0.5)(x)
    X = QConv1D(filters=32, kernel_size=3, kernel_quantizer="stochastic_ternary", bias_quantizer="ternary", name="conv1d_5")(x)
    x = QActivation("quantized_relu(3)")(x)
    X = QConv1D(filters=32, kernel_size=3, kernel_quantizer="stochastic_ternary", bias_quantizer="ternary", name="conv1d_6")(x)
    x = QActivation("quantized_relu(3)")(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = QDense(128, kernel_quantizer=quantized_bits(3), bias_quantizer=quantized_bits(3))(x)
    x = QActivation("quantized_relu(3)")(x)
    x = QDense(NUM_CLASSES, kernel_quantizer=quantized_bits(3), bias_quantizer=quantized_bits(3))(x)
    x = QActivation("quantized_bits(20, 5)")(x)
    x = Activation("softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    if load_weights and weights_f:
        model.load_weights(weights_f)

    print_qstats(model)
    return model

def UseNetwork(weights_f, load_weights=False, save_model = True):
  
    """Use Conv1D Model.
    Args:
        weights_f: weight file location.
        load_weights: load weights when it is True.
    """

    model = QConv1D_model(weights_f, load_weights)

    if not load_weights:
        model.fit(X_train, Y_train,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                verbose=2,
                validation_data=(X_val, Y_val),
                callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'])
    
    if weights_f:
        model.save_weights(weights_f)
    
    if save_model:
        model.save("./model/QConv1D.h5")
    

    score = model.evaluate(x_test_, y_test_, verbose=VERBOSE)
    print_qstats(model)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])


def ParserArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_weight", default="0",
                        help="""load weights directly from file.
                                0 is to disable and train the network.""")
    parser.add_argument("-w", "--weight_file", default=None)
    parser.add_argument("-s", "--save_model", default=None)
    a = parser.parse_args()
    return a


if __name__ == "__main__":
  args = ParserArgs()
  lw = False if args.load_weight == "0" else True
  UseNetwork(args.weight_file, load_weights=lw, args.save_model)