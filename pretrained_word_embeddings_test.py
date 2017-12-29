from __future__ import print_function
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

# 載入模型
model = load_model('embedding.h5')
# 讀取測試檔內容
PREDICT_TEXT_DATA_DIR = 'predict_data'
predict_path = os.path.join(PREDICT_TEXT_DATA_DIR, '1.txt')
f = open(predict_path, encoding='utf-8')
predict_text = f.read()
f.close()

# 轉成詞向量
texts=[predict_text]
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_predict = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of predict data tensor:', x_predict.shape)

# 預測
y_predict = model.predict(x_predict)
max_val = np.argmax(y_predict)
print('Category it belongs to : ',max_val)

print('y_predict : ',y_predict)


# {'rec.sport.baseball': 9, 'comp.windows.x': 5, 'comp.sys.mac.hardware': 4, 'comp.sys.ibm.pc.hardware': 3, 'sci.crypt': 11, 
#'sci.space': 14, 'rec.motorcycles': 8, 'talk.politics.guns': 16, 'misc.forsale': 6, 'talk.politics.mideast': 17, 'sci.med': 13, 
#'soc.religion.christian': 15, 'comp.graphics': 1, 'talk.politics.misc': 18, 'sci.electronics': 12, 'rec.autos': 7, 
#'rec.sport.hockey': 10, 'comp.os.ms-windows.misc': 2, 'alt.atheism': 0, 'talk.religion.misc': 19}