'''
三种模型的预测函数：
predImages:直接对1024维取欧式距离
pcaPred:PCA方法降维后取欧式距离
kmeansPred:k聚类方法降维后取欧式距离

使用方法如在90行之后的示例代码
'''
from keras.models import load_model
from keras import Model
import BruteForce as bf
import numpy as np
import Feature1024Extract as fe
from PIL import Image
def getFeature(modelpath,imglist):
    x=fe.img2array(imglist)
    model=load_model(modelpath)
    oriimgclass = model.predict(x)
    imgclass=[]
    for i in range(0,len(imglist)):
        max=0
        maxindex=0
        for j in range(0,10):
            if(oriimgclass[i][j] > max):
                max=oriimgclass[i][j]
                maxindex=j
        print((max,maxindex+1))
        imgclass.append(maxindex+1)
    imgclass=np.array(imgclass)
    model=Model(inputs=model.input,outputs=model.get_layer('fc-final').output)
    imgfeature=model.predict(x)
    return imgclass,imgfeature
def predImages(modelpath,imglist):
    imgclass,imgfeature=getFeature(modelpath,imglist)
    imgnum=imgclass.shape[0]
    imgs=[]
    for i in range(0,imgnum):
        data=np.load('Data/'+str(imgclass[i])+'-feature.npy')
        datalist=np.load('Data/'+str(imgclass[i])+'-list.npy')
        img = bf.dis(imgfeature[i],data,datalist)
        print(img)
        showimg(img)
        imgs.append(img)
    return imgs
def showimg(imglist):
    for img in imglist:
        image=Image.open(img)
        image.show()
def pcaPred(modelpath,imglist):
    imgclass,imgfeature=getFeature(modelpath,imglist)
    imgnum=imgclass.shape[0]
    imgs=[]
    for i in range(0,imgnum):
        arg=np.load('PCA/'+str(imgclass[i])+"-components.npy")
        lowfeature=np.dot(arg,imgfeature[i])
        print(lowfeature.shape)
        data=np.load('PCA/'+str(imgclass[i])+"-feature.npy")
        datalist=np.load('Data/'+str(imgclass[i])+'-list.npy')
        img=bf.dis(lowfeature,data,datalist)
        print(img)
        showimg(img)
        imgs.append(img)
    return imgs
def kmeansPred(modelpath,imglist):
    imgclass,imgfeature=getFeature(modelpath,imglist)
    imgnum=imgclass.shape[0]
    imgs=[]
    for i in range(0,imgnum):
        dim=imgfeature[i].shape[0]
        label=np.load('Kmeans/'+str(imgclass[i])+'-label.npy')
        data=np.load("Kmeans/"+str(imgclass[i])+'-feature.npy')
        datalist=np.load('Data/'+str(imgclass[i])+'-list.npy')
        classnum=data.shape[1]
        temp=[]
        temp_count=[]
        for j in range(0,classnum):
            temp.append(0)
            temp_count.append(0)
        for j in range(0,dim):
            temp[label[j]]=temp[label[j]]+imgfeature[i][j]
            temp_count[label[j]]=temp_count[label[j]]+1
        for j in range(0,classnum):
            temp[j]=temp[j]/temp_count[j]
        lowfeature=np.array(temp)
        img=bf.dis(lowfeature,data,datalist)
        print(img)
        showimg(img)
        imgs.append(img)
    return imgs
imglist=[]
img="n01923025_344.JPEG"
imglist.append(img)
pcaPred("InceResV2-final.npy",imglist)
