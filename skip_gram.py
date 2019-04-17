# from keras.layers import merge
# from keras.layers import Dense,Reshape
# from keras.layers.embeddings import Embedding
# from keras.models import Sequential
#
# vocab_size = 5000
# embed_size = 300
#
# word_model = Sequential()
# word_model.add(Embedding(vocab_size, embed_size, embeddings_initializer= "glorot_uniform", input_length=1))
# word_model.add(Reshape((embed_size, )))
#
# context_model = Sequential()
# context_model.add(Embedding(vocab_size, embed_size, embeddings_initializer= "glorot uniform", input_length=1))
# context_model.add(Reshape((embed_size, )))
#
# model = Sequential()
# model.add(merge([word_model, context_model], mode = "dot"))
# model.add(Dense(1, init= "glorot uniform", activation='sigmoid'))
# model.compile(loss = "mean_squared_error", optimizer="adam")
#
# from keras.preprocessing.text import *
# from keras.preprocessing.sequence import skipgrams
# text = "I love you and the sky"
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([text])
#
# word2id = tokenizer.word_index
# id2word = {v:k for k,v in word2id.items()}
#
# wids = [word2id[w] for w in text_to_word_sequence(text)]
# pairs, labels = skipgrams(wids, len(word2id))
# print(len(pairs), len(labels))
#
# for i in range(10):
from keras.layers import Dense,Dropout,SpatialDropout1D
from keras.layers import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np

np.random.seed(42)

input_flie = "train.txt"
vocab_size = 5000
embed_size = 100
num_fliter = 256
num_words = 3
batch_size = 64
epochs = 20

counter = collections.Counter()
fin = open(input_flie, "rb")
maxlen = 0
for line in fin:
    _,sent = line.strip().split()#"t"
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words)>maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(vocab_size)):
    word2index[word[0]] = wid +1
vocab_size = len(word2index) + 1
index2word = {v:k for k,v in word2index.items()}

xs, ys = [], []
fin = open(input_flie, "rb")
for line in fin:
    label, sent = line.strip().split()
    ys.append(int(label))
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    wids = [word2index[word] for word in words]
    xs.append(wids)

fin.close()
x = pad_sequences(xs, maxlen= maxlen)
y = np_utils.to_categorical(ys)

train_x, train_y, test_x, test_y = train_test_split(x, y, test_size=0.3, random_state= 42)
model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(Conv1D(filters=num_fliter, kernel_size= num_words, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=batch_size, epochs= epochs, validation_data=(test_x, test_y))