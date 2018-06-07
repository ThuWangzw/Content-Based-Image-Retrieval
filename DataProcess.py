import os
from PIL import Image
import numpy as np
import keras
from keras.applications.inception_v3 import preprocess_input
def raw_data_process(path,classNum):
    x=[]
    y=[]
    imglist=[]
    '''process data in the path and save as .npy'''
    for i in range(1,classNum+1):
        print(i)
        classpath=path+'/'+str(i)
        for img in os.listdir(classpath):
            image=Image.open(classpath+'/'+img)
            image=image.resize((299,299),Image.ANTIALIAS)
            image=np.array(image)
            x.append(image)
            temp=[]
            for j in range(1,11):
                temp.append(0)
            temp[i-1]=1
            y.append(temp)
            imglist.append(img)
    y=np.array(y)
    print(y)
    imglist=np.array(imglist)
    np.save("imglist.npy",imglist)
    np.save("y.npy",y)
def loadnpy():
    x=np.load("x.npy")
    y=np.load("y.npy")
    return x,y
