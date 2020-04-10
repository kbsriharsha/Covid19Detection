# !/Users/kbsriharsha/anaconda3/bin/python
# coding: utf-8
# @author: Bharat Sri Harsha karpurapu


"""
This program provides all the necessary preprocessing libraries
"""

# Importing libraries
import pandas as pd
import cv2
import numpy as np

def resize(img, width, height, interpolation=cv2.INTER_AREA):
    '''
    This function resizes the image
    '''
    return cv2.resize(img, (width, height), interpolation)


def images_from_folder(folder, label = 1):
    '''
    This function extracts all the images and resizes them to be used
    by MobileNet
    '''
    images = []
    labels = []
    for file in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,file))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize(image, 224, 224)
            images.append(image)
            labels.append(label)
    return images, labels
