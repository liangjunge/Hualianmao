from gensim.models import KeyedVectors
from  keras.layers import Dense, Dropout, SpatialDropout1D, GlobalAveragePooling1D
from keras.layers import Conv1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import gensim.models.word2vec
import numpy as np
np.random.seed(42)

input_file = "train.txt"
w2v_model = "google.bin"
vocab_size = 5000
embed_size = 300
num_filter = 256
num_words = 3
batch_size = 64
epochs = 10

counter  = collections.Counter()
fin = open(input_file, 'rb')
maxlen = 0
for line in fin:
    _, sent = line.strip().split()
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    #确定最大句长！
    if len(words)>maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid,word in enumerate(counter.most_common(vocab_size)):
    word2index[word[0]] = wid + 1
vocab_size = len(word2index) + 1
index2word = {v:k for k,v in word2index.items()}


#词向序列转换
xs, ys = [],[]
fin = open(input_file,'rb')
for line in fin:
    label, sent = line.strip().split()
    ys.append(label)
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    wids = [word2index[word] for word in words]
    xs.append(wids)
fin.close()
x = pad_sequences(xs, maxlen = maxlen)
y = np_utils.to_categorical(ys)

train_x, train_y ,test_x, test_y = train_test_split(x,y ,test_size=0.3, random_state= 42)

#加载谷歌词向量
word2vec = gensim.models.KeyedVectors.load_word2vec_format('google.bin',binary=True)
embedding_weights = np.zeros((vocab_size, embed_size))
for word, index in word2index.items():
    try:
        embedding_weights[index,:] = word2vec[word]
    except KeyError:
        pass

model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_length= maxlen, weights= embedding_weights, trainable=False))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(Conv1D(filters=num_filter, kernel_size=num_words, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size= batch_size, epoch =epochs, validation_data= (test_x, test_y))

score = model.evaluate(test_x, test_y)
print("test score:{.3f}, test loss: {.3f}",score[1], score[0])