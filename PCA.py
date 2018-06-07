'''
生成PCA相关的文件，即x-components与x-feature，后者是PCA方法得到的降维的feature,前者是降维的参数（即如何由1024维降为低维（这里是12维））
调用initPCA()即可进行以上操作
'''
import numpy as np
from sklearn.decomposition import PCA
def format(path):
    temp=[]
    feature=np.load(path)
    for i in range(0,5613):
        temp.append(feature[i].flatten().tolist())
        print(i)
    newfinal=np.array(temp)
    print(newfinal.shape)
    np.save("vgg19-linear.npy",newfinal)
def onePCA(oriFeaturePath,tarFeaturePath,tarComponentsPath,n_components):
    feature=np.load(oriFeaturePath)
    pca=PCA(n_components,copy=False)
    reduce=pca.fit_transform(feature)
    np.save(tarComponentsPath,pca.components_)
    np.save(tarFeaturePath,reduce)
    print(pca.explained_variance_ratio_)
def initPCA():
    for i in range(1,11):
        pcacomPath='PCA/'+str(i)+'-components.npy'
        pcafeaPath='PCA/'+str(i)+'-feature.npy'
        orifeature='Data/'+str(i)+'-feature.npy'
        onePCA(orifeature,pcafeaPath,pcacomPath,12)
initPCA()
