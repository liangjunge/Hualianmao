#-*-coding:utf-8-*-
#authorized by Niansheng
#2019.4.10 just for  play
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)
from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


#parameter
NB_epoch = 20
Batch_size = 128
Verbose =1
NB_classes = 10
Optimizer = SGD()
N_hidden = 128
Validation_split = 0.2

Reshaped = 784
(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, Reshaped).astype('float32')
X_test = X_test.reshape(10000, Reshaped).astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

y_train = np_utils.to_categorical(y_train, NB_classes)
y_test = np_utils.to_categorical(y_test, NB_classes)

model = Sequential()
model.add(Dense(N_hidden, input_shape= (Reshaped,)))
model.add(Activation('relu'))
model.add(Dense(N_hidden))
model.add(Activation('relu'))
model.add(Dense(NB_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer= Optimizer, metrics=[f1])
history = model.fit(X_train, y_train,epochs= NB_epoch,batch_size=Batch_size, verbose= Verbose, validation_split=Validation_split)

score = model.evaluate(X_test, y_test, verbose= Verbose)
print("Test Score", score[0])
print("Test accuracy", score[1])