from matplotlib.pyplot import get
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy ,AUC
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

SEED =1234
EMB_SIZE =100
SEQUENCE_LEN = 128
CLASS_NUM = 3
batch_size = 32



# loading
with open("sentiment_model2/tokenizer", 'rb') as handle:
    tokenizer = pickle.load(handle)

def load_data_test(data_name):
    data_test = pd.read_csv('test data/phase 2/final_data2.csv', index_col=None,  encoding="utf-8")
    print(data_test.isna().sum())
    data_test = data_test.dropna()
    x , y= np.asarray(data_test['review']), np.asarray(data_test['sentiment'])
    return x,y

def prepare_data(x , y ):
    sequence_docs = tokenizer.texts_to_sequences(x.astype(str))
    pad_sequence_docs = pad_sequences(sequence_docs, maxlen=SEQUENCE_LEN, padding='post')
    categorical_y = to_categorical(y, CLASS_NUM)
    return pad_sequence_docs, categorical_y

x_test , y_test= load_data_test('test')

test_pad_sequence, test_categorical_y = prepare_data(x_test , y_test)

model = load_model('sentiment_model2/model.h5'.format(emb_size=EMB_SIZE))

print("Evaluate on test data")
results = model.evaluate(test_pad_sequence, test_categorical_y, batch_size=batch_size)
print("test loss, test acc:", results)
predictions = model.predict(test_pad_sequence)

print(predictions)



def get_label(prediction):
    labels = ['-1', '0', '1']
    result = [labels[np.argmax(row)] for row in prediction]
    return result

labels = get_label(predictions)
df = pd.DataFrame(data = labels, columns = ['predict'])


df.to_csv("predict.csv")