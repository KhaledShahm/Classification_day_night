# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:23:13 2021

@author: Dodo_Shahm
"""

import cv2 # computer vision library
import helper as h

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = h.load_dataset(image_dir_training)

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = h.load_dataset(image_dir_test)


# Standardize all training images
STANDARDIZED_LIST = h.standardize(IMAGE_LIST)

# Standardize the test data
STANDARDIZED_TEST_LIST = h.standardize(TEST_IMAGE_LIST)

# Testing average brightness levels
# Look at a number of different day and night images and think about 
# what average brightness value separates the two types of images

# Find the average of the averages from all day and night images

night_brightness = []
day_brightness = []

for image in STANDARDIZED_LIST:
    
    if image[1] == 0:
        night_brightness.append(h.avg_brightness(image[0]))
    elif image[1] == 1:
        day_brightness.append(h.avg_brightness(image[0]))

avg_day_brightness = np.mean(day_brightness)
avg_night_brightness = np.mean(night_brightness)





# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Find all misclassified images in a given test set
MISCLASSIFIED = h.get_misclassified_images(STANDARDIZED_TEST_LIST, avg_day_brightness, avg_night_brightness)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

