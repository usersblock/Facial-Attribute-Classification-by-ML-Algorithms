import cv2
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


#Import image labels and paths to dataframe
def get_image_label_filepath_df(path):
    mypath = path
    filenames = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
    splitcolumns = [x.split('_')[0:3] + [mypath + x] for x in filenames if x.count('_') == 3]
    filecolumns = ['age','gender','race','file']
    filedf = pd.DataFrame(data = splitcolumns, columns = filecolumns)
    return filedf

def get_labels_npimages(df,y):
    X = df['file'].values
    y = df[y].astype(int).values
    img_container = []
    for i in X:
        img = cv2.imread(i)
        img_container.append(img)
    return img_container, y
    

#filedf = get_image_label_filepath_df('./Data/CroppedImages/')

#X,y = get_labels_npimages(filedf,'race')

#np.array(X).shape

#np.array(y).shape