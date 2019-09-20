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

num_train_samples = X_train.shape[0]

#==================PRUNE THE MODEL=====================

# Model reconstruction from JSON file
with open('./model/convmodrecnets_CNN2.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('./model/convmodrecnets_CNN2_0.5.wts.h5')

model.summary()



# Set up some params 
nb_epoch = 50     # number of epochs to train on
batch_size = 1024  # training batch size
initial_sparsity = 0.0
final_sparsity = 0.9
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * nb_epoch
print("End step: ", end_step)

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                 final_sparsity=final_sparsity,
                                                 begin_step=2000,
                                                 end_step=end_step,
                                                 frequency=100)
}

l = tf.keras.layers
dr = 0.5 # dropout rate (%)
pruned_model = models.Sequential()
pruned_model.add(Reshape([1]+in_shp, input_shape=in_shp))
pruned_model.add(sparsity.prune_low_magnitude(Conv2D(64, (2, 8), padding='valid', data_format="channels_first", 
                                                     activation="relu", name="conv1", kernel_initializer='glorot_uniform'), **pruning_params))
pruned_model.add(Dropout(dr))
pruned_model.add(sparsity.prune_low_magnitude(Conv2D(32, (1, 32), padding='valid', data_format="channels_first", 
                                                     activation="relu", name="conv2", kernel_initializer='glorot_uniform'), **pruning_params))
pruned_model.add(Dropout(dr))
pruned_model.add(Flatten())
pruned_model.add(sparsity.prune_low_magnitude(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"), **pruning_params))
pruned_model.add(Dropout(dr))
pruned_model.add(sparsity.prune_low_magnitude(Dense(len(classes), kernel_initializer='he_normal', name="dense2" ), **pruning_params))
pruned_model.add(Activation('softmax'))
pruned_model.compile(loss='categorical_crossentropy', optimizer='adam')

pruned_model.summary()



callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0),
             sparsity.UpdatePruningStep(),
             sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

history = pruned_model.fit(X_train, Y_train,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           verbose=1,
                           validation_data=(X_test, Y_test),
                           callbacks=callbacks)

score = pruned_model.evaluate(X_test, Y_test, verbose=0)



#Save the model
pruned_model = sparsity.strip_pruning(pruned_model)
pruned_model.summary()

# Save the model architecture
with open('convmodrecnets_CNN2-pruned.json', 'w') as f:
    f.write(pruned_model.to_json())
    
# Save the weights
model.save_weights('convmodrecnets_CNN2_0.5.wts-pruned.h5')






