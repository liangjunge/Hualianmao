from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #  CONV=>RELU=>POOL
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #  CONV=>RELU=>POOL
        model.add(Conv2D(50, kernel_size=5,  border_mode = 'same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #Flatten
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        #Softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return  model

Nb_epoch = 30
Batch_size = 128
Verbose = 1
Optimizer = Adam()
Validation_split = 0.2
Img_rows, Img_cols = 28, 28
Nb_calsses = 10
Input_shape = (1, Img_rows, Img_cols)

(X_train,y_train), (X_test, y_test) = mnist.load_data()
K.set_image_dim_ordering("th")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

X_train = X_train[:, np.newaxis, :,:]
X_test = X_test[:, np.newaxis, :,:]

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")


y_trian = np_utils.to_categorical(y_train, Nb_calsses)
y_test = np_utils.to_categorical(y_test, Nb_calsses)

model = LeNet.build(input_shape=Input_shape, classes= Nb_calsses)
model.compile(loss="categorical_crossentropy", optimizer=Optimizer, metrics=["accuracy"])

history = model.fit(X_train, y_trian, batch_size=Batch_size, verbose= Verbose,validation_split= Validation_split)
score = model.evaluate(X_test, y_test, verbose= Verbose)
print("test score:" ,score[0])
print("test accuracy:", score[1])

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history["val_acc"])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(["train", "test"], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()