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
from skimage.color import rgb2gray


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)

@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)



def sobelFilterHSV(imagesArray):
    """This function takes as input an array of pictures and returns two arrays. The first array has the 
    Sobel filter apply to each picture channel and the second one is applied on the image's hsv. The images are in color"""

    #filter apply to hue, saturation, value
    filterHSV = []
    for image in imagesArray:
        image2 = rescale_intensity(1 - sobel_hsv(image))
        filterHSV.append(image2)
    return np.array(filterHSV)

def sobelChannel(imagesArray):
    filterChannel = [] 
    for image in imagesArray:
        image1 = rescale_intensity(1 - sobel_each(image))
        filterChannel.append(image1)
        
    return np.array(filterChannel)
    
def robertsEdges(imagesArray):
    from skimage import color
    """returns two numpy arrays with pictures in black and white and with the edges define using Roberts filter
    on the first one and the Sobel filter on the second one. The images are in black and white"""
    roberts_ = []
    for image in imagesArray:
        grayImg = rgb2gray(image)
        edge_roberts = filters.roberts(grayImg)
        roberts_.append(edge_roberts)
       
    return np.array(roberts_)

def sobelEdges(imagesArray):
    sobel_ = []
    for image in imagesArray:
        grayImg = rgb2gray(image)
        edge_sobel = filters.sobel(grayImg)
        sobel_.append(edge_sobel)
    return np.array(sobel_)

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

def gammaCorrection(imagesArray):
    """
    histogram equalizer 
    """
    gamm = []
    for image in imagesArray:
        gamma_corrected = exposure.adjust_gamma(image, 2) 
        gamm.append(gamma_corrected)
        
    return np.array(gamm)

def logCorrection(imagesArray):
    log = []
    for image in imagesArray:
        logarithmic_corrected = exposure.adjust_log(image, 1)
        log.append(logarithmic_corrected)
    return np.array(log)


def histogramEqualizer(imagesArray):
    """
    returns 3 numpy arrays one is with rescale, equalize histogram, adapthis
    
    """
    from skimage import color
    
    hist = []
    
    for image in imagesArray:
        image = color.rgb2gray(image)
        # Equalization
        img_eq = exposure.equalize_hist(image)
        # Adaptive Equalization
        hist.append(img_eq)
        
        
    return np.array(hist)

def intesityEqualizer(imagesArray):
    intensity = []
    for image in imagesArray:
        image = color.rgb2gray(image)
        p2, p98 = np.percentile(image, (2, 98))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        intensity.append(img_rescale)
    return np.array(intensity)

def adaptistEqualizer(imagesArray):
    adapthist = []
    for image in imagesArray:
        image = color.rgb2gray(image)
        mg_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
        adapthist.append(img_adapteq)
        
    return np.array(adapthist)

def images_extractor(mypath):
    import pandas as pd
    from os import listdir
    import numpy as np

    filenames = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
    splitcolumns = [x.split('_')[0:3] + [x] for x in filenames if x.count('_') == 3]
    filecolumns = ['age','gender','race','file']
    filedf = pd.DataFrame(data = splitcolumns, columns = filecolumns)
    filedfnona = filedf.dropna()
    filedfnona['age']  = filedfnona['age'].astype(int)
    filedfnona['race'] = filedfnona['race'].astype(int)
    filedfnona['gender'] = filedfnona['gender'].astype(int)

    images = []
    for img in filedfnona['file']:
        image = plt.imread(mypath + img)
        images.append(image)
    return images

 








