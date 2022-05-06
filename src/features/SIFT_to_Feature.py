import sys
sys.path.insert(0,'..\\data\\')
import make_dataset_beta as md
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans
from scipy.spatial.distance import cdist
import pandas as pd
import cv2

# Get Image Descriptors, which are a combination of points on an image and the description of surrounding pixels.

def get_descriptors(nparrays,nfeatures):
    sift = cv2.SIFT_create(nfeatures = nfeatures)
    container = []
    for i in nparrays:
        img_bw = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        keypoint,descriptor = sift.detectAndCompute(img_bw,None)
        container.append([keypoint,descriptor])
    return container

#For a collection of image area descriptions, get the Kmeans of n clusters. This will be what future images are compared to.

def get_vocab(descriptors,n):
    descriptor_container = []
    for i in descriptors:
        if i[1] is None:
            continue
        for j in i[1]:
            descriptor_container.append(j)
    vocab = kmeans(descriptor_container,n)
    return vocab

# For each image, get its descriptors. For each descriptor, get the closest Kmean descriptor in vocab and add 1 to its index in a histogram.
# Return a histogram per image. This histogram will be passed as a feature for modeling.

def descriptor_to_vocab(nparrays,vocab):
    sift = cv2.SIFT_create()
    container = []
    for i in nparrays:
        img_bw = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        keypoint,descriptor = sift.detectAndCompute(img_bw,None)
        if descriptor is not None:
            dist = cdist(descriptor,vocab[0],'euclidean')
            bin_assignment = np.argmin(dist,axis = 1)
        else:
            bin_assignment = []
        image_feats = np.zeros(len(vocab[0]))
        for j in bin_assignment:
            image_feats[j] += 1
        container.append(image_feats)
    return container

# Normalizes histograms from images so that they may be used in ML inputs

def normalize_histograms(histarray):
    histarray = np.array(histarray)
    feats_norm_div = np.linalg.norm(histarray,axis = 1)
    for i in range(0,histarray.shape[0]):
        histarray[i] = histarray[i]/feats_norm_div[i]
    return histarray

# Pipeline for SIFT to histogram features per image
# Returns a dataframe for training and testing datasets
'''
path - path to the image directory
column - what type of label to return
nvocab - how many vectors will appear in the descriptor 'vocabulary'
test_size - percentage of images that will be used as tests
random_state - seed for randomization
n_features - For each image, return at most n descriptors
'''

def SIFT_to_Features(path,column,nvocab = 200,test_size = 0.33,random_state = 42,n_features = 100):
    filedf = md.get_image_label_filepath_df('./Data/CroppedImages/')
    X,y = md.get_labels_npimages(filedf,column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    descriptors = get_descriptors(X_train_np[:,1],n_features)
    vocab = get_vocab(descriptors[:100],nvocab)
    # Get histogram values from X_train descriptors
    histarraytrain = descriptor_to_vocab(X_train_np[:,1],vocab)
    # Get histogram values from X_test descriptors
    histarraytest = descriptor_to_vocab(X_test_np[:,1],vocab)
    # Normalize train histograms
    normalizehisttrain = normalize_histograms(histarraytrain)
    # Normalize test histograms
    normalizehisttest = normalize_histograms(histarraytest)
    traindf = pd.DataFrame((X_train_np[:,0],normalizehisttrain,y_train_np))
    testdf = pd.DataFrame((X_test_np[:,0],normalizehisttest,y_test_np))
    traindf = traindf.transpose()
    testdf = testdf.transpose()
    return [traindf,testdf]

#train,test = SIFT_to_Features('PATH','race')
#train.head()
#test.head()