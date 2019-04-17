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
glove_model = 'glove.txt'
vocab_size = 5000
embed_size  = 100
batch_size = 64
epochs = 10

counter = collections.Counter()
fin = open(input_file, 'rb')
maxlen = 0

for line in fin:
    _, sent = line.strip().split()
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for index, word in enumerate(counter.most_common(vocab_size)):
    word2index[word[0]] = index + 1
vocab_size = len(word2index)
index2id = {k:v for v,k in word2index.items()}
index2id[0] = "_UNK_"

x = np.zeros((W.shapes))

