





#-*-coding:utf-8-*-
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from  keras.layers import Dense, Activation
from keras.optimizers import SGD
from  keras.utils import np_utils
np.random.seed(1671)
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
(train_X, train_y),(test_X, test_y) = mnist.load_data()

"""
a显示mnist图片，1行10列，结果如Figure1.jpg所示
"""
fig, ax = plt.subplots(
    nrows=1,
    ncols=10,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(10):
    img = train_X[train_y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""
b预处理先输出
"""
print('train_X.shape', train_X.shape)
print('train samples', train_X.shape[0])
print('test samples', test_X.shape[0])
# train_x.shape (60000, 28, 28)
# train samples 60000
# test samples 10000
"""
reshape 两种模式
"""
# train_X = np.reshape(train_X, (60000, 784))
# print('train_X.shape',train_X.shape)
train_X = train_X.reshape(60000, 784)
test_X = test_X.reshape(10000, 784)
print('train_X.shape',train_X.shape)
# # (60000, 784)

"""
c归一化数据,转换成float类型
"""

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X /= 255
test_X /= 255

train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)

"""
d 搭建3层神经网络
"""
model = Sequential()
model.add(Dense(128, input_shape = (784, )))#f隐层节点个数，128换成5/10/20/50/100
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

"""
e 分割训练、验证、测试集
"""


train_X_ = train_X[:42000]
train_y_ = train_y[:42000]
print('train_X.shape', train_X_.shape)
print('train samples', train_X_.shape[0])
# train_X.shape (42000, 784)
# train samples 42000
train_test_X = train_X[-9000:]
train_test_y = train_y[-9000:]
print('test_X.shape', train_test_X.shape)
print('test samples', train_test_X.shape[0])

val_X = train_X[42000:51000]
val_y = train_y[42000:51000]
print('val_X.shape', val_X.shape)
print('val samples', val_X.shape[0])


early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
history = model.fit(train_X_, train_y_, batch_size= 128, epochs= 20, verbose = 1, validation_data = (val_X, val_y),) #callbacks= [early_stopping])
score = model.evaluate(test_X, test_y, verbose = 1)

"""混淆矩阵"""


import pandas as pd


y_pred = np.argmax(model.predict(test_X),axis=1)
y_true = np.argmax(test_y, axis = 1)
cm = confusion_matrix(y_true,y_pred ) #混淆矩阵
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, 10)