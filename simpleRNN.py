from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
import numpy as np

fin = open('train.txt', 'rb')
lines = []
for line in fin:
    line = line.strip().split()
    line = line .decode("ascii","ignore")
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()
text = " ".join(lines)

chars = set([c for c in text])
# vocab = []
# for word in chars:
#基于字的
nb_chars = len(chars)
char2index = dict((c,i) for i, c in enumerate(chars))
index2char = dict((i,c) for i, c in enumerate(chars))

Seqlen = 10
step =1

"""
创建输入和标签文本
"""
input_chars = []
label_chars = []

for i in range(0, len(text)- Seqlen, step):
    #0-10/1-11
    input_chars.append(text[i:1+ Seqlen])
    label_chars.append(text[i+ Seqlen])


x = np.zeros((len(input_chars), Seqlen, nb_chars), dtype = np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype =np.bool)
"""
字符级的，先是字，再是字母
"""
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        x[i, j,char2index[ch]] = 1
        y[i, j, char2index[ch]] =1
    y[i, char2index[label_chars[i]]] = 1


hidden_size = 128
batch_size = 128
num_iterations = 25
num_epochs_per_iteration = 1
num_preds_pre_epoch = 100

model =Sequential()
model.add(SimpleRNN(hidden_size, return_sequences= False, input_shape = (Seqlen, nb_chars), unroll = True))
model.add(Dense(chars))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop")

for  iteration in range(num_iterations):
    print("="*50)
    print("Iteration #: %d"%iteration)
    model.fit(x,y, batch_size= batch_size, epochs= num_epochs_per_iteration)

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed: %s" %(test_chars))
    print(test_chars, end = '')
    for i in range(num_epochs_per_iteration):
        test_x = np.zeros((1, Seqlen, nb_chars))
        for i ,ch in enumerate(test_chars):
            test_x[0, i, char2index[ch]] = 1
        pred = model.predict(test_x, verbose=  0)[0]
        y_pred = index2char[np.argmax(pred)]
        print(y_pred, end= '')

        test_chars = test_chars[1:] + y_pred
    print()
    