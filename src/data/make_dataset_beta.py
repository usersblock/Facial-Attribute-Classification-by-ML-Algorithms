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
    filedf = filedf.astype({'age': 'int32','gender':'int32','race':'int32'})
    return filedf

def get_labels_npimages(df,y):
    if y == 'age':
        df['age']=np.where((df.age<3), 1, df.age)
        df['age']=np.where(((df.age>=3) & (df.age<6)), 2, df.age)
        df['age']=np.where(((df.age>=6) & (df.age<9)), 3, df.age)
        df['age']=np.where(((df.age>=9) & (df.age<12)), 4, df.age)
        df['age']=np.where(((df.age>=12) & (df.age<21)), 5, df.age)
        df['age']=np.where(((df.age>=21) & (df.age<36)), 6, df.age)
        df['age']=np.where(((df.age>=36) & (df.age<51)), 7, df.age)
        df['age']=np.where(((df.age>=51) & (df.age<80)), 8, df.age)
        df['age']=np.where((df.age>=80), 9, df.age)
    X = df['file'].values
    y = df[y].astype(int).values
    img_container = []
    for i in X:
        img = cv2.imread(i)
        img_container.append([i,img])
    return img_container, y
    

#filedf = get_image_label_filepath_df('./Data/CroppedImages/')

#X,y = get_labels_npimages(filedf,'race')

#np.array(X).shape

#np.array(y).shape