'''
在这个文件中调用init_feature()，会产生Data中x-feature.npy文件与x-list.npy文件，
其中，x-feature是使用网络训练5613张图片得到的5613个feature，x-list只是记录这些图片的名字，主要是为了与x-feature中的顺序相对应，方便找到feature对应哪个图片
'''
from keras.models import load_model
from keras import Model
from PIL import Image
import numpy as np
import os
def img2array(imgliat):
    x=[]
    for img in imgliat:
        image=Image.open(img)
        image=image.resize((299,299),Image.ANTIALIAS)
        image=np.array(image)
        x.append(image)
    x=np.array(x)
    return x
def ExtractFeatures(modelpath,imglist,feature=None):
    base_model=load_model(modelpath)
    model=Model(inputs=base_model.input,outputs=base_model.get_layer('fc-final').output)
    x=img2array(imglist)
    print("begin extract features")
    y=model.predict(x,verbose=1)
    if(feature is not None):
        np.save(feature,y)
    return y
def getClasses(modelpath,imglist):
    model=load_model(modelpath)
    x=img2array(imglist)
    y=model.predict(x)
    return y
def init_feature():
    path='raw_data'
    classNum=10
    for i in range(1,classNum+1):
        print(i)
        classpath=path+'/'+str(i)
        imglist=[]
        for img in os.listdir(classpath):
            imglist.append(classpath+'/'+img)
        ExtractFeatures('InceResV2-final.npy',imglist,'Data/'+str(i)+'-feature.npy')
        np.save('Data/'+str(i)+'-list.npy',np.array(imglist))
