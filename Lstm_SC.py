from keras.layers import  Dense, Activation, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import  matplotlib.pyplot as plt
import  numpy as np
import nltk
import os

Data_dir = '/data'
maxlen = 0
word_fre = collections.Counter()
num_res = 0
ftrain = open(os.path.join(Data_dir, "train.txt"), 'rb')
for line in ftrain:
    labels, sentence = line.strip().split()
    words = nltk.word_tokenize(sentence.decode('utf-8').lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_fre[word] += 1
ftrain.close()
# vocab_size = len(word_fre)

max_features = 2000
max_sentence_length = 40

vocab_size = min(max_features, len(word_fre)) + 2
word2index = {x[0]:i+2 for i, x in enumerate(word_fre.most_common(max_features)) }
word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {k:v for v,k in word2index.items()}

x = np.empty((num_res), dtype=list)
y = np.empty((num_res),)
ftrain = open(os.path.join(Data_dir, "train.txt"), 'rb')
for line in ftrain:
    labels, sentence = line.strip().split()
    words = nltk.word_tokenize(sentence.decode('utf-8').lower())
    seqs = []
    for word in words:
        #wids = [words2index[word] for word in words]
        if word2index.__contains__(word):
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
        i = 0
        x[i] = seqs
        y[i] = int(labels)
        i += 1

ftrain.close()
x = sequence.pad_sequences(x, maxlen=)
train_x, train_y, test_x, test_y = train_test_split(x, y, test_size= 0.2, random_state=42)

embedding_size = 128
hidden_layer_size = 64
batch_size = 32
num_epochs = 10

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length = max_sentence_length))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(LSTM(hidden_layer_size, dropout= 0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss= 'binary_crossentropy', optimizer= 'adam',metrics= ['accuracy'])

history = model.fit(train_x, train_y, batch_size= batch_size, epochs= num_epochs, validation_data=(test_x, test_y))
plt.plot(211)
plt.title("Acc")
plt.plot(history.history["acc"], color = 'g', label = "train")
plt.plot(history.history['val_acc'], color = 'r', label = "Validation")
plt.legend(loc = "best")
plt.show()
"""
随机找几个句子，打印出RNN的预测结果，标签和实际的句子
"""

score, acc = model.predict(test_x, test_y, batch_size = batch_size, verbose= 1)
print("test score: %.3f, accuracy : %.3f"%(score, acc))

for i in range(5):
    idx = np.random.randint(len(test_x))
    test_x = test_x[idx].reshape(1, 40)
    ylabel = test_y[idx]
    y_pred = model.predict(test_x)[0][0]
    sent = " ".join([index2word[x] for x in test_x[0].tolist() if x != 0])
    print("%.0ft%dt%s"%(y_pred, ylabel, sent))
