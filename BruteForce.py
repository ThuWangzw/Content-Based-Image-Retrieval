'''
feature:某张图片提取出来的feature
x:该图片对应的class的所有图片的feature
name：class中所有图片的名字

返回值：10张最相近的图片的名字
'''
import numpy as np
def dis(feature,x,name):
    dis=[]
    num=x.shape[0]
    for i in range(0,num):
        temp=[]
        temp.append(np.sqrt(np.sum(np.square(x[i]-feature))))
        temp.append(name[i])
        dis.append(temp)
    for i in range(0,num-1):
        for j in range(0,num-1-i):
            if(dis[j][0]>dis[j+1][0]):
                temp=dis[j]
                dis[j]=dis[j+1]
                dis[j+1]=temp
    img=[]
    for i in range(0,10):
        img.append(dis[i][1])
    return img
