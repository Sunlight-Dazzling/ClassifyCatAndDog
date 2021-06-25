# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:38:06 2021

@author: Administrator
"""

from keras.applications import VGG19
conv_base = VGG19(weights = 'imagenet',include_top = False,input_shape=(150, 150, 3))
conv_base.summary()


import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# 数据集分类后的目录
base_dir = 'E:\\Cat_And_Dog\\kaggle\\cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale = 1. / 255)
batch_size = 20
def extract_features(directory, sample_count):
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count))
    generator = datagen.flow_from_directory(directory, target_size = (150, 150), 
                                            batch_size = batch_size,
                                            class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        #把图片输入VGG16卷积层，让它把图片信息抽取出来
        features_batch = conv_base.predict(inputs_batch)
        #feature_batch 是 4*4*512结构
        features[i * batch_size : (i + 1)*batch_size] = features_batch
        labels[i * batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count :
            #for in 在generator上的循环是无止境的，因此我们必须主动break掉
            break
        return features , labels
#extract_features 返回数据格式为(samples, 4, 4, 512)
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4* 512))

from keras import models
from keras import layers
from keras import optimizers
#构造我们自己的网络层对输出数据进行分类
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim = 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr = 2e-5), loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(train_features, train_labels, epochs = 30, batch_size = 20, 
                    validation_data = (validation_features, validation_labels))

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Train_acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Trainning and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
