


SEED =1234
EMB_SIZE =100
SEQUENCE_LEN = 128
CLASS_NUM = 3

# !pip install fasttext
# !pip install hazm
# !pip install --upgrade gensim

import fasttext 
import hazm
import pandas as pd
import pickle
import numpy as np
import gensim


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout
from tensorflow.keras.layers import  LSTM, Bidirectional,GRU

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy ,AUC
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.utils import plot_model


import matplotlib.pyplot as plot

FT_EMBEDDING_PATH = 'sentiment_model/cc.fa.300.bin'

# ft_embedding = fasttext.load_model('/content/drive/MyDrive/Karimi/cc.fa.300.bin')
# len(ft_embedding.words)

def get_tokenized_data(data_name):
    x , y= load_data(data_name)
    tokenized_x =  [hazm.word_tokenize(sent) for sent in x]
    return tokenized_x,y

def load_data_test(data_name):
    data_test = pd.read_csv('phase3/test.csv', index_col=None,  encoding="utf-8")
    print(data_test.isna().sum())
    data_test = data_test.dropna()
    x , y= np.asarray(data_test['review']), np.asarray(data_test['sentiment'])
    return x,y

def load_data_train(data_name):
    data_train = pd.read_csv('phase3/train.csv', index_col=None,  encoding="utf-8")
    print(data_train.isna().sum())
    data_train = data_train.dropna()
    x , y= np.asarray(data_train['review']), np.asarray(data_train['sentiment'])
    return x,y

def load_data_valid(data_name):
    data_valid = pd.read_csv('phase3/valid.csv', index_col=None,  encoding="utf-8")
    print(data_valid.isna().sum())
    data_valid = data_valid.dropna()
    x , y= np.asarray(data_valid['review']), np.asarray(data_valid['sentiment'])
    return x,y




x_test , y_test= load_data_test('test')
x_train , y_train= load_data_train('train')
x_val , y_val= load_data_valid('valid')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

# Save tokenizer
with open("sentiment_model2/tokenizer", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

from gensim.models import FastText
from gensim.test.utils import common_texts

def train_fasttext_emb(corpus):
    sentences = [[word for word in hazm.word_tokenize(text)] for text in corpus]
    emb_model = FastText(sentences, vector_size=100, min_count=1)
    emb_model.build_vocab(sentences)
    emb_model.train(sentences, total_examples=len(corpus), epochs=10)  
    emb_model.save(FT_EMBEDDING_PATH)
    return emb_model

# Train the embedding
emb_model = train_fasttext_emb(x_train)

# load the pretrained model
#emb_model = FastText.load(EMBEDDING_PATH)

print(emb_model.wv.most_similar('عالی'))

def get_emb_matrix(emb_model_):
    embedding_matrix = np.random.random((len(tokenizer.word_index)+1, emb_model_.vector_size))
    pas = 0
    for word,i in tokenizer.word_index.items():
        try:
            embedding_matrix[i] = emb_model_.wv[word]
        except:
            print(word)
            pas+=1
    return embedding_matrix

embedding_matrix = get_emb_matrix(emb_model)

def prepare_data(x , y ):
    sequence_docs = tokenizer.texts_to_sequences(x.astype(str))
    pad_sequence_docs = pad_sequences(sequence_docs, maxlen=SEQUENCE_LEN, padding='post')
    categorical_y = to_categorical(y, CLASS_NUM)
    return pad_sequence_docs, categorical_y


train_pad_sequence, train_categorical_y = prepare_data(x_train , y_train)
val_pad_sequence, val_categorical_y = prepare_data(x_val , y_val)
test_pad_sequence, test_categorical_y = prepare_data(x_test , y_test)

#from sklearn.utils import class_weight
import sklearn
import numpy as np
class_w = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_train) , y_train)
class_w_dic = {}
for i in range(len(class_w)):
    class_w_dic[i] = class_w[i]
    
class_w_dic

class_w_dic={0:70, 1:20, 2:10}

temp_model = Sequential()
temp_model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False))
temp_model.add(Bidirectional(GRU(100, return_sequences=True, name='lstm_layer')))
temp_model.add(Dropout(0.2))
temp_model.add(Dense(50, activation="relu", kernel_regularizer='l2'))
temp_model.add(Bidirectional(GRU(50)))
temp_model.add(Dropout(0.1))
temp_model.add(Dense(3, activation='softmax', kernel_regularizer='l2'))

#Config
METRICS = [
    categorical_accuracy,
    tf.keras.metrics.AUC(name='auc'),
    #tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
temp_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=[METRICS])

temp_model.summary()

batch_size = 32
epochs = 20
history = temp_model.fit(train_pad_sequence, train_categorical_y,
                         batch_size=batch_size, epochs=epochs, 
                         class_weight=class_w_dic,
                         validation_data=(val_pad_sequence, val_categorical_y), shuffle=True)

import matplotlib.pyplot as plt
#plt.plot(history.history['loss'] , label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("LOSS")
plt.show

#FIX EMB 100
batch_size = 256
epochs = 13
history = temp_model.fit(train_pad_sequence, train_categorical_y,
                         batch_size=batch_size, epochs=epochs, 
                         class_weight=class_w_dic,
                         validation_data=(val_pad_sequence, val_categorical_y) , 
                         shuffle=True)

print("Evaluate on test data")
results = temp_model.evaluate(test_pad_sequence, test_categorical_y, batch_size=batch_size)
print("test loss:{loss}, test acc:{acc}, test auc: {auc}".format( loss = results[0],acc = results[1] ,auc=results[2] ))

# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = temp_model.predict(test_pad_sequence[:3])
print("predictions shape:", predictions.shape)
preds = np.argmax(predictions, axis = 1)
print(x_test[:3] , preds)

import os

if not os.path.exists('sentiment_model2'):
    os.makedirs('sentiment_model2')
temp_model.save('sentiment_model2/model.h5'.format(emb_size=EMB_SIZE),
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)

from tensorflow.keras.models import load_model

model = load_model('sentiment_model2/model.h5'.format(emb_size=EMB_SIZE))

print("Evaluate on test data")
results = model.evaluate(test_pad_sequence, test_categorical_y, batch_size=batch_size)
print("test loss, test acc:", results)
predictions = model.predict(test_pad_sequence)