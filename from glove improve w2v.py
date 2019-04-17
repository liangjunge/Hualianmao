from keras.layers.embeddings import Embedding
from keras.models import Model,Sequential
from keras.layers import SpatialDropout1D
import numpy as np

glove_model = 'glove.txt'
word2emb = {}

vocab = 5000
embed_size = 300

words = []
#maxlen
maxlen = 0
for word in words:
    if len(word)>0:
        maxlen = len(word)

fglove = open(glove_model,'rb')
for line in fglove:
    cols = line.strip().split()
    word = cols[0]
    embedding = np.array(cols[1:], dtype='float32')
    word2emb[word] = embedding
fglove.close()

embedding_weights = np.zeros((vocab, embed_size))
for word, index in word2emb.items():
    try:
        embedding_weights[index,: ] = word2emb[word]
    except KeyError:
        pass
"""SpatialDropout1D与Dropout类似，但它断开的是整个1D特征图，而不是单个神经元，
如果一张特征图的相邻像素之间有很强的相关性，那么普通的Dropout无法将其正则化输出，否则就会
导致明显的学习率下降，这种情况下，Spar可以帮助提高特征图之间的独立性。
"""
model = Sequential()
model.add(Embedding(vocab, embed_size, input_length=maxlen,weights= [embedding_weights], trainable= False))
model.add(SpatialDropout1D(0.2))