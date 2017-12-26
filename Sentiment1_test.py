from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
from keras.datasets import imdb 

MAX_SENTENCE_LENGTH = 400
##### 自己輸入
from keras.models import load_model
model=load_model("Sentiment1.h5")
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with open('./Sentiment1_training.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))

## 準備數據
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word_index["PAD"] = 0
word_index["UNK"] = 1
#index2word = {v:k for k, v in word_index.items()}

INPUT_SENTENCES = ['I love it.','It is so boring.']
label2word = {1:'正面', 0:'負面'}
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in INPUT_SENTENCES:
    seq = []
    words = nltk.word_tokenize(sentence.lower())
    for word in words:
        if word in word_index:
            seq.append(word_index[word])
        else:
            seq.append(word_index['UNK'])
    XX[i] = seq
    i+=1
#x_new = [[word_index[w] for w in sentence if w in word_index]]
XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
print(model.predict(XX))
labels = [int(round(x[0])) for x in model.predict(XX) ]
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
