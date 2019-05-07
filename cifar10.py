from keras.datasets import cifar10
from  keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y),(test_x, test_y) = cifar10.load_data()
# print('train_x.shape', train_x.shape)
# print('train samples', train_x.shape[0])
# print('test samples', test_x.shape[0])
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /= 255
test_x /= 255
train_y_ = []
train_x_ = []
"""
获取前3个类别的数据训练
"""
for j in range(len(train_y)):
     if train_y[j] == [0]or train_y[j] == [1] or train_y[j] == [2]:
         train_y_.append(train_y[j])
         train_x_.append(train_x[j])

train_x_ = np.array(train_x_)#.reshape((32,32,3))
train_y_ = np.array(train_y_)
# print(train_y_[1])
# print(train_x_[1])
print('train_x_.shape', train_x_.shape)
print('train samples', train_x_.shape[0])
# print(train_y_[:50])

train_x1 = train_x_[:10500]
train_y1 = train_y_[:10500]

val_x1 = train_x_[-2250:]
val_y1 = train_y_[-2250:]

train_test_x1 = train_x_[10500:12750]
train_test_y1 = train_y_[10500:12750]



test_x1 = []
test_y1 = []
for j in range(len(test_y)):
     if test_y[j] == [0] or test_y[j] == [1] or test_y[j] == [2]:
         test_y1.append(test_y[j])
         test_x1.append(test_x[j])
test_x1 = np.array(test_x1)#.reshape((32,32,3))
test_y1 = np.array(test_y1)
# print(test_x1[1])
# print(test_y1[1])
# print(test_y1[:10])
print('test_x_.shape', test_x1.shape)
print('test samples', test_x1.shape[0])

# print('train_x.shape', train_x_.shape)
# print('train samples', train_x_.shape[0])
# print('test samples', test_x.shape[0])
train_y1 = np_utils.to_categorical(train_y1, 3)
val_y1 = np_utils.to_categorical(val_y1, 3)
test_y1 = np_utils.to_categorical(test_y1, 3)





Input_shape =(32, 32, 3)

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', input_shape=Input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64)) #dense层
model.add(Activation('relu'))
model.add(Dense(512)) #dense层
model.add(Activation('relu'))
model.add(Dense(512)) #dense层
model.add(Activation('relu'))
# model.add(Dense(512)) #dense层
# model.add(Activation('relu'))
# model.add(Dense(512)) #dense层
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer= Adam(), metrics=['accuracy'])
history = model.fit(train_x1, train_y1, batch_size=128, verbose=1, validation_data= (val_x1, val_y1), epochs= 20)

score = model.evaluate(test_x1, test_y1, batch_size=128, verbose= 1)
print('test loss', score[0])
print('test acc', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc = 'upper left')
plt.show()


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc = 'upper left')
plt.show()
