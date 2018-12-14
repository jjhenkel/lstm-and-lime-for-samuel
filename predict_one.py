import numpy as np
import labels as L

import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.contrib.keras as keras
import tensorflow as tf

from keras import backend as K

K.set_learning_phase(0)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.engine import Layer, InputSpec, InputLayer

from keras.models import Model, Sequential, load_model

from keras.layers import Dropout, Embedding, concatenate
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Conv2D, MaxPool2D, ZeroPadding1D
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.layers import Concatenate, Dot, Merge, Multiply, RepeatVector
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import SimpleRNN, LSTM, GRU, Lambda, Permute

from keras.layers.core import Reshape, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.metrics import top_k_categorical_accuracy

import keras.metrics

def top_3_accuracy(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=3)
keras.metrics.top_3_accuracy = top_3_accuracy

import pickle


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 100
MAX_NUMBER_WORDS = 136085

texts = []
for line in sys.stdin:
    texts.append(line.strip())

for t in texts:
  assert len(t.split(' ')) <= MAX_SEQUENCE_LENGTH

tokenizer = None
with open('/app/tokenizer.pkl', 'rb') as wordf:
  tokenizer = pickle.load(wordf)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# print('Shape of data tensor:', data.shape)

model = load_model('/app/best-1.h5')

answers = model.predict(data, batch_size=32)

get = lambda i: [ l[0] for l in L.LABELS.items() if l[1][0] == i ][0]

for j,answer in enumerate(answers):
  decoded = sorted([ (get(i).replace('$', ''), answer[i]) for i in range(0, len(answer)) ], key=lambda x: -x[1])
  print('{}'.format(
    '|'.join([ '{},{:.8f}'.format(*r) for r in decoded ])
  ))

import gc; gc.collect()
