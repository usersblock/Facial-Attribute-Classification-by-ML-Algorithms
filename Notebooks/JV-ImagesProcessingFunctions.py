# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import os
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from skimage.exposure import rescale_intensity
from skimage.data import camera
from skimage.util import compare_images
from skimage import exposure
from skimage import data, img_as_float
from skimage import exposure

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

def sobelFilterTypes(imagesArray):
    """
    This function takes as input an array of pictures and returns two arrays. The first array has the 
    Sobel filter apply to each picture channel and the second one is applied on the image's hsv. The images are in color
    """
    #filter apply to pictures channel
    filterChannel = []
    #filter apply to hue, saturation, value
    filterHSV = []
    for image in imagesArray:
        image1 = rescale_intensity(1 - sobel_each(image))
        image2 = rescale_intensity(1 - sobel_hsv(image))
        filterChannel.append(image1)
        filterHSV.append(image2)
    return np.array(filterChannel), np.array(filterHSV)
        

def robertsSobelEdges(imagesArray):
    from skimage import color
    """
    returns two numpy arrays with pictures in black and white and with the edges define using Roberts filter
    on the first one and the Sobel filter on the second one. The images are in black and white
    """
    roberts_ = []
    sobel_ = []
    for image in imagesArray:
        grayImg = color.rgb2gray(image)
        edge_roberts = filters.roberts(grayImg)
        edge_sobel = filters.sobel(grayImg)
        roberts_.append(edge_roberts)
        sobel_.append(edge_sobel)
    return np.array(roberts_), np.array(sobel_)

def exposure_(imagesArray):
    from skimage import exposure
    """
    This fucntion increases the exposure of each picture and it returns a numpy
    array of the modified images. The images are in color
    """
    exp = []
    for image in imagesArray:
        image_eq = exposure.equalize_hist(image)
        exp.append(image_eq)
    return np.array(exp)

def gammaLogaritmictCorrection(imagesArray):
    """
    histogram equalizer 
    """
    gamm = []
    log = []
    for image in imagesArray:
        gamma_corrected = exposure.adjust_gamma(image, 2)
        logarithmic_corrected = exposure.adjust_log(image, 1)
        gamm.append(gamma_corrected)
        log.append(logarithmic_corrected)
    return np.array(gamm), np.array(log)

def histogramEqualizer(imagesArray):
    """
    returns 3 numpy arrays one is with rescale, equalize histogram, adapthis
    
    """
    
    
    
    intensity = []
    hist = []
    adapthist = []
    for image in imagesArray:
        p2, p98 = np.percentile(image, (2, 98))
        
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

        # Equalization
        img_eq = exposure.equalize_hist(image)

        # Adaptive Equalization
        img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
        
        intensity.append(img_rescale)
        hist.append(img_eq)
        adapthist.append(img_adapteq)
        
    return np.array(intensity), np.array(hist), np.array(adapthist)