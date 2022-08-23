import os
import time
import sys

import cv2
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from scipy.special import inv_boxcox
from scipy.stats import boxcox

from config import SAMPLE_LENGTH, BATCH_SIZE, denoised_img, write_feature_path
    
def read_image(X_path):
    '''
    Reads in an image as an array, from a given path.

    Inputs:
    --------------
    X_path : str
        Path to the image.

    Outputs:
    --------------
    X_image : numpy array
        Image found at X_path.
    '''

    X_image = cv2.imread(X_path, 0)
    
    return X_image

def create_sample_ROI(image, sample_length=128, verbose=False):
    '''
    Create the center of the ROI by randomly selecting points on
    the x- and the y- axis. If the center is too close to the edge
    (meaning that the ROI will go off the page), the selection 
    trial is run again until a valid coordinate has been chosen.
    
    Inputs
    --------------
    image : array
        Original image.
        
    sample_percentage : int
        Changes the size of the ROI. Larger numbers mean larger ROI.
        Between 1-98. ROIs become exponentially more difficult to find
        as sample_percentage approaches 100. So 95 is the cutoff.
        
    Outputs
    --------------
    roi : np.array
        array of the ROI of the image in question.
        
    x_random : int
        Randomly selected x-coordinate for ROI.
        
    y_random : int
        Randomly selected y-coordinate for ROI.
    '''
    x_length = image.shape[0]
    y_length = image.shape[1]
    
    x_random = random.randint(1, x_length+1)
    y_random = random.randint(1, y_length+1)

    while ( 
        (y_random + sample_length) > y_length
        or (y_random - sample_length) < 0 
    ):
        if verbose:
            print(
                "Cannot create ROI: y_random exceeds ROI center limits.\n"
                "A new y_random will be chosen.\n\n"
            )
        y_random = random.randint(1, y_length+1)

    while ( 
        (x_random + sample_length) > x_length
        or (x_random - sample_length) < 0 
    ):
        if verbose:
            print(
                "Cannot create ROI: x_random exceeds ROI center limits.\n"
                "A new x_random will be chosen.\n\n"
            )
        x_random = random.randint(1, x_length+1)
        
    roi = image[
        x_random - sample_length : x_random + sample_length,
        y_random - sample_length : y_random + sample_length
    ]
    
    return roi

def execute(img_manifest, BATCH_SIZE, SAMPLE_LENGTH, write_feature_path, ROI=True):
    ftr = pd.DataFrame()
    ftrs_tmp = []
    batch = BATCH_SIZE
    sample_length = SAMPLE_LENGTH
    j = 0
    print('start tabulating.')
    for (i, elem0) in enumerate(img_manifest):
        ftr = pd.DataFrame()
        if ROI:
            img = create_sample_ROI(read_image(elem0), sample_length=sample_length//2)
        else:
            img = read_image(elem0)
        ttl_dim = img.shape[0]*img.shape[1]
        
        # Turn image into row
        tmp = np.squeeze(img.reshape([1, ttl_dim]))
        ftrs_tmp.append(tmp)

        if i % batch == 0 and i != 0:
            ftr = ftr.append(ftrs_tmp, ignore_index=True)
            output_path_modified = os.path.abspath(os.path.join(write_feature_path, 'ftr_'+str(BATCH_SIZE)+'_'+str(j)+'_.csv'))
            ftr.to_csv(output_path_modified)
            j += 1
            ftr = None
            ftrs_tmp = []
            print(f'{i+1} elements of {len(img_manifest)} processed.')
    
    print(ftrs_tmp)
    if ftrs_tmp:
        ftr = ftr.append(ftrs_tmp, ignore_index=True)
        output_path_modified = os.path.abspath(os.path.join(write_feature_path, 'ftr_'+str(BATCH_SIZE)+'_'+str(j)+'_.csv'))
        ftr.to_csv(output_path_modified)
        ftrs_tmp = []
    print(f'{len(img_manifest)} elements of {len(img_manifest)} processed.')

################################################################################
################################################################################
def tabulate():
    print('Tabulating images.')

    # Get the manifest of the features (images).
    img_manifest = []

    print('appending')
    for root, _, files in os.walk(denoised_img):
        for (i, f) in enumerate(files):
            img_manifest.append(os.path.join(root, f))
            print(f'{i} files')
    print('sorting')
    img_manifest.sort()

    # Tabulate the images.
    execute(img_manifest, BATCH_SIZE, SAMPLE_LENGTH, write_feature_path)

    print('Tabulating images finish.')