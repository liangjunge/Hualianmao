from keras.datasets import cifar10
from  keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

(train_x, train_y),(test_x, test_y) = cifar10.load_data()
print('train_x.shape', train_x.shape)
print('train samples', train_x.shape[0])
print('test samples', test_x.shape[0])

classes = 10
train_y = np_utils.to_categorical(train_y, classes)
test_y = np_utils.to_categorical(test_y, classes)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /= 255
test_x /= 255

Input_shape =(32, 32, 3)

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=Input_shape))
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
model.fit(train_x, train_y, batch_size=128, verbose=1, validation_split= 0.2, epochs= 5)

score = model.evaluate(test_x, test_y, batch_size=128, verbose= 1)
print('test loss', score[0])
print('test acc', score[1])


model_json = model.to_json()
open('cifar_model.json', 'w').write(model_json)
model.save_weights('cifar_weights', overwrite= True)
