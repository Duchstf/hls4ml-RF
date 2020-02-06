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

#Input shape and number of classes
in_shape = list(X_train.shape[1:])
classes = mods
num_classes = len(classes)

print("X_train shape: ", X_train.shape)
print("Number of classes: ", num_classes)

#==================DEFINE THE MODEL=====================

def prune_Conv1D(final_sparsity, initial_sparsity = 0.0, begin_step = 0, frequency = 100, version = ""):
    # Set up some params 
    nb_epoch = 50     # number of epochs to train on
    batch_size = 1024  # training batch size 
    num_train_samples = X_train.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * nb_epoch
    print("End step: ", end_step)

    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                     final_sparsity=final_sparsity,
                                                     begin_step=begin_step,
                                                     end_step=end_step,
                                                     frequency=100)
    }

    l = tf.keras.layers
    dr = 0.5 # dropout rate (%)
    pruned_model = tf.keras.Sequential([
            sparsity.prune_low_magnitude(l.Conv1D(128, 3, padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform',input_shape=in_shape), **pruning_params),
            sparsity.prune_low_magnitude(l.Conv1D(128, 3, padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform'), **pruning_params),
            l.MaxPool1D(2),
            sparsity.prune_low_magnitude(l.Conv1D(64, 3, padding='valid', activation="relu", name="conv3", kernel_initializer='glorot_uniform'), **pruning_params),
            sparsity.prune_low_magnitude(l.Conv1D(64, 3, padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform'), **pruning_params),
            l.Dropout(dr),
            sparsity.prune_low_magnitude(l.Conv1D(32, 3, padding='valid', activation="relu", name="conv5", kernel_initializer='glorot_uniform'), **pruning_params),
            sparsity.prune_low_magnitude(l.Conv1D(32, 3, padding='valid', activation="relu", name="conv6", kernel_initializer='glorot_uniform'), **pruning_params),
            l.Dropout(dr),
            l.MaxPool1D(2),
            l.Flatten(),
            sparsity.prune_low_magnitude(l.Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"), **pruning_params),
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
                               validation_data=(X_val, Y_val),
                               callbacks=callbacks)

    score = pruned_model.evaluate(X_test, Y_test, verbose=0)

    print("Test loss: ", score)

    #Save the model
    pruned_model = sparsity.strip_pruning(pruned_model)
    pruned_model.summary()

    # Save the model architecture
    print_model_to_json(pruned_model, './model/Conv1D-{}.json'.format(str(final_sparsity) + version))

    # Save the weights
    pruned_model.save_weights('./model/Conv1D-{}.h5'.format(str(final_sparsity) + version))

#==================TRAIN=====================
prune_Conv1D(initial_sparsity = 0.0, final_sparsity = 0.85, begin_step = 2500, version = "v1", frequency = 150)
