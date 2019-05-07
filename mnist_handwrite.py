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
model.add(Dense(100, input_shape = (784, )))#f隐层节点个数，128换成5/10/20/50/100
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
"""
# f绘训练误差和验证误差
# """
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'], loc = 'upper left')
plt.show()
print("Test score", score[0])
print("Test accuracy", score[1])

# history = model.fit_generator(train_X_, train_y_, batch_size= 128, epochs= 20, verbose = 1, validation_data = (val_X, val_y),) #callbacks= [early_stopping])
# score = model.evaluate_generator(test_X, test_y,verbose=1)
# prediction=model.predict_generator(validation_generator,verbose=1)
"""混淆矩阵"""
# y_pred = np.argmax(model.predict(test_X),axis=1)
# y_true = np.argmax(test_y, axis = 1)
#
# cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.green)#Blues
#
# # make various adjustments to the plot
# plt.tight_layout()
# plt.colorbar()
# tick_marks = np.arange(10)
# plt.xticks(tick_marks, range(10))
# plt.yticks(tick_marks, range(10))
# plt.xlabel("Predicted")
# plt.ylabel(("True"))
# plt.show()
#
# def cm_plot(y, yp):
#   from sklearn.metrics import confusion_matrix #导入混淆矩阵函数
#   cm = confusion_matrix(y, yp) #混淆矩阵
#   import matplotlib.pyplot as plt #导入作图库
#   plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
#   plt.colorbar() #颜色标签
#   for x in range(len(cm)): #数据标签
#     for y in range(len(cm)):
#       plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
#   plt.ylabel('True label') #坐标轴标签
#   plt.xlabel('Predicted label') #坐标轴标签
#   return plt


y_pred = np.argmax(model.predict(test_X),axis=1)
y_true = np.argmax(test_y, axis = 1)
Tr = []
fa = []
for i in range(len(y_pred)):
    if y_pred[i] != y_true[i]:
        print('in ',i,'label is')
        print('pred label is',y_pred[i])
        print('true lable is', y_true[i])#         fa.append(y_pred[i])
#         Tr.append(y_true[i])
# plt.imshow(Tr[1])
# plt.show()

# y_true.tolist().append(10)
# y_pred.tolist().append([10])
# y_true=np.array(y_true)
# if 10 in y_true:
#     print(True)
cm = confusion_matrix(y_true,y_pred ) #混淆矩阵
a=[0]*10
cm=np.row_stack((cm,a))
b=[0]*11
cm=np.column_stack((cm,b))
plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens
plt.colorbar() #颜色标签
zongshu=[]
for x in range(len(cm)): #数据标签
    for y in range(len(cm)):
        zongshu.append(cm[x,y])
zongshu=sum(zongshu)
zhengque=[]
for x in range(len(cm)): #数据标签
    for y in range(len(cm)):
        zhengque.append(cm[x,x])
zhengque=sum(zhengque)

for x in range(len(cm)-1): #数据标签
    for y in range(len(cm)-1):
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='bottom')
        plt.annotate(round(cm[x, y]/zongshu*100,2), xy=(x, y), horizontalalignment='center', verticalalignment='top')
        # plt.annotate(round(cm[x,x]/sum(cm[x,:])*100,2),xy=(10,x))
for x in range(len(cm)-1): #数据标签
    for y in range(len(cm)-1):
        # plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='bottom')
        # plt.annotate(round(cm[x, y]/zongshu*100,2), xy=(x, y), horizontalalignment='center', verticalalignment='top')
        plt.annotate(round(cm[x,x]/sum(cm[x,:])*100,1),xy=(10,x), horizontalalignment='center', verticalalignment='bottom')
        plt.annotate(round((sum(cm[x,:])-cm[x, x]) / sum(cm[x, :]) * 100, 1), xy=(10, x), horizontalalignment='center',
                     verticalalignment='top')
        plt.annotate(round(cm[y,y] / sum(cm[:,y]) * 100, 1), xy=(y,10), horizontalalignment='center',
                     verticalalignment='bottom')
        plt.annotate(round((sum(cm[:,y])-cm[y,y]) / sum(cm[:, y])*100, 1), xy=(y, 10), horizontalalignment='center',
                     verticalalignment='top')
        plt.annotate(round((zhengque/zongshu) * 100, 1), xy=(10, 10),
                     horizontalalignment='center',
                     verticalalignment='bottom')
        plt.annotate(round((1-zhengque/zongshu) * 100, 1), xy=(10, 10),
                     horizontalalignment='center',
                     verticalalignment='top')

plt.title("Confusion Matrix")
plt.ylabel('True label') #坐标轴标签
plt.xlabel('Predicted label') #坐标轴标签
plt.show()



