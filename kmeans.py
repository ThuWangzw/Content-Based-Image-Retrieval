'''
与PCA.py类似，x-feature为降维后的feature，x-label为降维的相关参数
'''
from sklearn.cluster import KMeans
import numpy as np
def kmeans_feature(k,orifeaturepath,labelpath,tarfeaturepath):
    feature=np.load(orifeaturepath).T
    dim=feature.shape[0]
    imgnum=feature.shape[1]
    kmeans=KMeans(k,max_iter=1000,verbose=1,n_init=5).fit(feature)
    label=kmeans.labels_
    print(label.shape)
    np.save(labelpath,label)
    newfeature=[]
    label_count=[]
    for j in range(0,k):
        label_count.append(0)
    for j in range(0,dim):
        label_count[label[j]]=label_count[label[j]]+1
    print(label_count)
    for i in range(0,imgnum):
        temp=[]
        for j in range(0,k):
            temp.append(0)
        for j in range(0,dim):
            temp[label[j]]=temp[label[j]]+feature[j][i]
        for j in range(0,k):
            temp[j]=temp[j]/label_count[j]
        newfeature.append(temp)
    newfeature=np.array(newfeature)
    np.save(tarfeaturepath,newfeature)
def init_kmeans():
    for i in range(1,11):
        newfeaturePath='Kmeans/'+str(i)+'-feature.npy'
        labelPath='Kmeans/'+str(i)+'-label.npy'
        orifeature='Data/'+str(i)+'-feature.npy'
        kmeans_feature(10,orifeature,labelPath,newfeaturePath)
init_kmeans()
