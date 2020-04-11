# !/Users/kbsriharsha/anaconda3/bin/python
# coding: utf-8
# @author: Bharat Sri Harsha karpurapu

"""
This program trains the models created and produces the output
"""
# Importing libraries
import os
import numpy as np
import pandas as pd
import keras
import shutil
import tensorflow as tf
import itertools

from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Dropout, Activation, Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing import image
from keras.metrics import categorical_crossentropy
from keras.layers.convolutional import *
from keras.applications import imagenet_utils
from keras.optimizers import adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import models as mod
import helper_functions as hp

# Importing the image data
covid_features, covid_labels = hp.images_from_folder("/".join(os.getcwd().split("/")[:-1] + ["data","covid"]))
#"/".join(os.getcwd().split("/")[:-1] + ["data","covid"])
noncovid_features, noncovid_labels = hp.images_from_folder("/".join(os.getcwd().split("/")[:-1] + ["data","covid"]))

# Preparing the full dataset
images = covid_features + noncovid_features
labels = covid_labels + noncovid_labels

# Converting into numpy arrays
data = np.asarray(images)
labels = np.asarray(labels)

# Preprocessing the image data and converting the labels into
# categorical representation
data = hp.preprocess_input(data)
labele = keras.utils.to_categorical(labels)

# Splitting the data into train and test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

# data generator (Image Augmentation Process)
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


# Model Build
model = mod.mobilenet(num_classes = 2)
# Freezing all the layers except the last 23 layers
for layer in model.layers[:-23]:
        layer.trainable = False
print(model.layers)

# Model Compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.adamax(lr=1e-2),  # we can take big lr here because we fixed first layers
    metrics=['accuracy']  # report accuracy during training
)
model.summary()

# Parameters
# lr = 1e-2
# batch size = 24

# Model Fitting
model.fit_generator(data_generator.flow(trainX,trainY,24),
                        steps_per_epoch=len(trainX) / 24,
                        epochs = 50,
                        validation_data=(testX, testY),
	                    validation_steps=len(testX) / 24,
                        verbose=1)

# 
