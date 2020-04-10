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

import models
