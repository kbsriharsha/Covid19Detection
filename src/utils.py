# !/Users/kbsriharsha/anaconda3/bin/python
# coding: utf-8
# @author: Bharat Sri Harsha karpurapu

"""
This program provides all the necessary utils funcctions
"""

# Importing libraries
import pandas as pd
import cv2
import numpy as np


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def preprocess_predict(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1)
    return x
