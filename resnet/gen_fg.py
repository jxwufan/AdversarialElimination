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
import foolbox
from tqdm import tqdm
import pickle

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)

sess = tf.Session()
keras.backend.set_session(sess)

filepath = "./resnet56/cifar10_ResNet38v1_model.206.h5"
model = load_model(filepath)

fmodel = foolbox.models.KerasModel(model, bounds=(0, 1),
         preprocessing=(x_train_mean, 1))

y_train_label = y_train
y_test_label = y_test

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


attack  = foolbox.attacks.FGSM(fmodel)

x_adv_train = []
for i in tqdm(range(len(x_train))):
    image = x_train[i]
    label = int(y_train_label[i])
    x_adv = attack(image, label, epsilons=1,  max_epsilon=0.3)
    if x_adv is None:
        x_adv = image
    x_adv_train.append(x_adv)
x_adv_train = np.array(x_adv_train)

x_adv_test = []
for i in tqdm(range(len(x_test))):
    image = x_test[i]
    label = int(y_test_label[i])
    x_adv = attack(image, label, epsilons=1,  max_epsilon=0.3)
    if x_adv is None:
        x_adv = image
    x_adv_test.append(x_adv)
x_adv_test = np.array(x_adv_test)

scores = model.evaluate(x_test - x_train_mean, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

scores = model.evaluate(x_adv_test - x_train_mean, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

pickle.dump([x_adv_train, x_adv_test], open("./fg.pkl", "wb"))

