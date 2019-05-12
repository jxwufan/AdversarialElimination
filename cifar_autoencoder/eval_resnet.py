from __future__ import print_function
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import os
import pickle

def eval(dataset):
    fg_train, fg_test, fg_train_decoded, fg_test_decoded = pickle.load(open(dataset+'_decoded.pkl'))
    print("*********************************************************" +dataset +"_test" + "*********************************************************")
    scores = model.evaluate(fg_test - x_train_mean, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    print("*********************************************************" +dataset+ "_test_decode" + "*********************************************************")
    scores = model.evaluate(fg_test_decoded - x_train_mean, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])



num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
#TODO add mean
x_train -= x_train_mean
x_test -= x_train_mean
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

sess = tf.Session()
keras.backend.set_session(sess)

filepath = "./resnet56/cifar10_ResNet38v1_model.206.h5"
model = load_model(filepath)

eval('df')
eval('fg')
eval('bim')
eval('cifar10')

