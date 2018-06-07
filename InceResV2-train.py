'''
主要功能：训练改编成10维的Inception-Reanet-V2网络，将模型存储到InceResV2-final.npy中，InceResV2-ori.npy是训练过程中产生的另一个model，也可以用来预测，效果不如前者好。
'''
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model

from keras.optimizers import SGD
from PIL import Image
import numpy as np
from numpy.random import randint
def generate_batch_data_random(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen / batch_size
    while (True):
        i = randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
base_model=InceptionResNetV2(weights='imagenet',include_top=False)
base_model.save_weights("InceResV2-baseweights.npy")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu',name='fc-final')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
x=np.load("x.npy")
y=np.load("y.npy")

test_x=x[:100]
test_y=y[:100]
# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit_generator(generate_batch_data_random(x,y,32),steps_per_epoch=190,epochs=5,verbose=1)
model.save("InceResV2-ori.npy")
# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。
for layer in model.layers[:779]:
   layer.trainable = False
for layer in model.layers[779:]:
   layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 我们继续训练模型，这次我们训练最后两个 Inception block
# 和两个全连接层
model.fit_generator(generate_batch_data_random(x,y,32),steps_per_epoch=190,epochs=5,verbose=1)
model.save("InceResV2-final.npy")
