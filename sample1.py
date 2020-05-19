import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Training
from keras.models import Model, Input
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import keras as k
from keras_contrib.layers import CRF

#Fitting
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt



df = pd.read_csv("ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
df.head()
data = df[['sentence_idx','word','tag']]

print(data.head(20))
print(df['tag'].value_counts())

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
          
getter = SentenceGetter(data)
sentences = getter.sentences
print(sentences[1:3])

words = list(set(data["word"].values))
words.append("ENDPAD")
n_words = len(words); n_words

tags = list(set(data["tag"].values))
n_tags = len(tags); n_tags

max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
# print(word2idx["Families"])


maxlen = max([len(s) for s in sentences])
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=maxlen, sequences=X, padding="post",value=n_words - 1)
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]



# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#TRaining the model

input = Input(shape=(30,))
word_embedding_size = 150

# Embedding Layer
model = Embedding(input_dim=n_words, output_dim=word_embedding_size, input_length=30)(input)

# BI-LSTM Layer
model = Bidirectional(LSTM(units=word_embedding_size, 
                           return_sequences=True, 
                           dropout=0.5, 
                           recurrent_dropout=0.5, 
                           kernel_initializer=k.initializers.he_normal()))(model)
model = LSTM(units=word_embedding_size * 2, 
             return_sequences=True, 
             dropout=0.5, 
             recurrent_dropout=0.5, 
             kernel_initializer=k.initializers.he_normal())(model)

# TimeDistributed Layer
model = TimeDistributed(Dense(n_tags, activation="relu"))(model)  

# CRF Layer
crf = CRF(n_tags)

out = crf(model)  # output
# print("CRF", crf)
# print("OUT", out)
model = Model(input, out)

#FIT in the model
#Optimiser 
adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

# Compile model
model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])

model.summary()

# history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=5,
#                     validation_split=0.1, verbose=1)

# Saving the best model only
filepath="ner-bi-lstm-td-model-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the best model
history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=20, validation_split=0.1, verbose=1, callbacks=callbacks_list)

# from math import nan

# words = list(set(data["word"].values))
# n_words = len(words)

# tags = []
# for tag in set(data["tag"].values):
#     if tag is nan or isinstance(tag, float):
#         tags.append('unk')
#     else:
#         tags.append(tag)
# n_tags = len(tags)
# # from future.utils import iteritems

# word2idx = {w: i for i, w in enumerate(words)}
# tag2idx = {t: i for i, t in enumerate(tags)}
# idx2tag = {v: k for k, v in iteritems(tag2idx)}