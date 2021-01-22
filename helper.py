# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:28:33 2021

@author: Dodo_Shahm
"""
import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["day", "night"]
    
    # Iterate through each type folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))
    
    return im_list

def standardize_input(image):
    
    # Resize image so that all "standard" images are the same size 600x1100
    standard_im = cv2.resize(image, dsize=(1100, 600), interpolation=cv2.INTER_CUBIC)
    return standard_im

# Examples: 
# encode("day") should return: 1
# encode("night") should return: 0

def encode(label):        
    if label == "day":
        numerical_val = 1
    elif label == "night":
        numerical_val = 0
        
    return numerical_val


def standardize(image_list):
    
    # Empty image data array
    standard_list = []
    
    for item in image_list:
        image = item[0]
        label = item[1]
        
        # Standardize the image 
        standardized_im = standardize_input(image)
        
        # Create a numerical label 
        binary_label = encode(label)
        
        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, binary_label))
        
    return standard_list


def avg_brightness(rgb_image):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Add up all the pixel values in the V channel 
    sum_brightness = np.sum(hsv[:,:,2])
    
    # Calculate the average brightness using the area of the image
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]
    area = height*width
    # and the sum calculated above
    avg = sum_brightness/area
    return avg

def colors_green_blue(rgb_image):
    # Finds amount of green and blue there is in picture
    # At night there are little green and blue colors
    
    
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Color selection, green and blue
    lower_hue = np.array([90,0,0]) 
    upper_hue = np.array([135,255,255])
    # Define the masked area in HSV space
    mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

    plt.imshow(mask_hsv, cmap = 'gray')
    # Add up all the pixel values in the V channel
    sum_color = np.sum(mask_hsv)
    
    area = 600*1100.0  # pixels
    
    # Colors that are green and blue, 0 non, 255 all
    amount = ((sum_color/area)/255)*100
    
    # Return value between 0 and 100
    return amount

def estimate_label(rgb_image, avg_brightness_day, avg_brightness_night):
    
    # Extract average brightness feature from an RGB image 
    avg = avg_brightness(rgb_image)
    
    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0

    # Extract amount color of green and blue in image
    amount = colors_green_blue(rgb_image)
    
    # Set the value of a threshold that will separate day and night images
    threshold = avg_brightness_night + (avg_brightness_day - avg_brightness_night)/2
    #print(threshold)
    ## Return the predicted_label (0 or 1) based on whether the avg is 
    # above or below the threshold
    
     # Set the amount of green and blue wee expect in daytime
    threshold_2 = 6
    

    if(avg > threshold-5 and amount > threshold_2):
        # if the average brightness is above the threshold value, we classify it as "day"
        predicted_label = 1
    # else, the pred-cted_label can stay 0 (it is predicted to be "night")

    return predicted_label    

# Constructs a list of misclassified images given a list of test images and their labels
def get_misclassified_images(test_images, avg_brightness_day, avg_brightness_night):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im, avg_brightness_day, avg_brightness_night)

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels
