# scikit-learn 須升級至 0.19
# pip install -U scikit-learn
# 在 python 執行 nltk.download(), 下載 data
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
from keras.models import load_model

## 探索數據分析(EDA)
# 計算訓練資料的字句最大字數
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
index2word = {v:k for k, v in word_index.items()}
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
# 讀取訓練資料，將每一單字以 dictionary 儲存
with open('./Sentiment1_training.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word_index:
                seqs.append(word_index[word])
            else:
                seqs.append(word_index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1

# 字句長度不足補空白        
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# 資料劃分訓練組及測試組
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# 模型構建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
model = Sequential()
# 加『嵌入』層
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
# 加『LSTM』層
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
# binary_crossentropy:二分法
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

# 模型訓練
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

# 預測
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('預測','真實','句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
    
# 模型存檔
model.save('Sentiment1.h5')  # creates a HDF5 file 'model.h5'
    
##### 自己輸入測試
INPUT_SENTENCES = ['I love it.','It is so boring.', 'I love it althougn it is so boring.']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
# 轉換文字為數值
i=0
for sentence in  INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word_index:
            seq.append(word_index[word])
        else:
            seq.append(word_index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# 預測，並將結果四捨五入，轉換為 0 或 1
labels = [int(round(x[0])) for x in model.predict(XX) ]
label2word = {1:'正面', 0:'負面'}
# 顯示結果
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
