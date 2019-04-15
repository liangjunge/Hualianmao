from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
from keras.datasets import cifar10
from  keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

Num_to_augment = 5

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

print("Augment training set images...")
datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    zoom_range = 0.2,
    horizontal_flip = True ,
    fill_mode = 'nearest')

xtas, ytas = [], []
for i in range(train_x.shape[0]):
    num_aug = 0
    x = train_x[i]
    x = x.reshape((1, ) + x.shape)
    for x_aug in datagen.flow(x, batch_size=1, save_to_dir= 'preview', save_prefix= 'cifar', save_format= 'jpeg'):
        if num_aug >= Num_to_augment:
            break
        xtas.append(x_aug[0])
        num_aug += 1
Input_shape =(32, 32, 3)
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=Input_shape))
model.add(Activation('relu'))
model.add((Conv2D(32, (3,3))))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add((Conv2D(64, (3,3), padding='same')))
model.add(Activation('relu'))
model.add((Conv2D(64, (3,3))))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer= Adam(), metrics=['accuracy'])

datagen.fit(train_x)
histort = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128), samples_per_epoch = train_x.shape[0], verbose= 1)
score = model.evaluate(test_x, test_y, batch_size = 128, verbose = 1)
print("Test score", score[0])
print("Test accuracy", score[1])