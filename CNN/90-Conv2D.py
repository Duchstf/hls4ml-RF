#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:00:45 2019
@author: duc_hoang
"""
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import tensorboard
import tensorflow.keras as keras

from tensorflow.keras.models import model_from_json
from tensorflow_model_optimization.sparsity import keras as sparsity
import keras.models as models

from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

import tempfile
logdir = tempfile.mkdtemp()

def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'w')
    jsonString = keras_model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

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
n_train = int(n_examples * 0.8)
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

#Add empty dimension at the end for CONV input layer
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

#Take input shape
in_shp = list(X_train.shape[1:])
print("X_train shape: ", X_train.shape)
classes = mods

num_train_samples = X_train.shape[0]

#==================PRUNE THE MODEL=====================

# Set up some params 
nb_epoch = 50     # number of epochs to train on
batch_size = 100  # training batch size
initial_sparsity = 0.0
final_sparsity = 0.9
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * nb_epoch
print("End step: ", end_step)

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                 final_sparsity=final_sparsity,
                                                 begin_step=0,
                                                 end_step=end_step,
                                                 frequency=100)
}

l = tf.keras.layers
dr = 0.5 # dropout rate (%)
pruned_model = tf.keras.Sequential([
        sparsity.prune_low_magnitude(l.Conv2D(64, (2, 8), padding='valid', data_format="channels_last",
						activation="relu", name="conv1", kernel_initializer='glorot_uniform',input_shape=in_shp), **pruning_params),
        l.Dropout(dr),
        sparsity.prune_low_magnitude(l.Conv2D(32, (1, 32), padding='valid', data_format="channels_last", 
                                                     activation="relu", name="conv2", kernel_initializer='glorot_uniform'), **pruning_params),
        l.Dropout(dr),
        l.Flatten(),
        sparsity.prune_low_magnitude(l.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"), **pruning_params),
        l.Dropout(dr),
        sparsity.prune_low_magnitude(l.Dense(len(classes), kernel_initializer='he_normal', name="dense2" ), **pruning_params),
        l.Activation('softmax') 
        ])
        
pruned_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])

pruned_model.summary()



callbacks = [sparsity.UpdatePruningStep(),
             sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

history = pruned_model.fit(X_train, Y_train,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           verbose=1,
                           validation_data=(X_test, Y_test),
                           callbacks=callbacks)

score = pruned_model.evaluate(X_test, Y_test, verbose=0)

print("Test loss: ", score)

#Save the model
pruned_model = sparsity.strip_pruning(pruned_model)
pruned_model.summary()

# Save the model architecture
print_model_to_json(pruned_model, 'Conv2D-90.json')
    
# Save the weights
pruned_model.save_weights('Conv2D-90.h5')

