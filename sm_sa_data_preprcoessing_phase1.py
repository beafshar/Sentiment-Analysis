

# !pip install -q hazm
# !pip install -q parsivar
# !pip install -q clean-text[gpl]

from hazm import *
import hazm
from cleantext import clean

import plotly.express as px
import plotly.graph_objects as go

from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import fileinput

import argparse
import sys
import re

import emoji
import string

data = pd.read_csv("test data/test.csv")

lemmatizer = Lemmatizer()

def remove_space(text):
    translation_src = "\u200b\u200d_-,.\n"
    translation_dst = "\u200c\u200f     "
    text = re.sub(translation_src, ' ', text)
    text = re.sub(translation_dst, ' ', text)
    text = re.sub('\u200f', ' ', text)
    return text

def remove_punct(text):
    text = re.sub('[0-9]+', '', text)
    text = re.sub('[۱-۹]+', '', text)
    return text

def emoji(text): 
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_shadda(text):
     text = re.sub('\u0651', '', text) #ARABIC SHADDA
     text = re.sub('\u064a', '', text) #ARABIC LETTER YEH  
     text = re.sub('\u0649', '', text) #ARABIC LETTER ALEF MAKSURA  
     text = re.sub('\u0652', '', text) #ARABIC SUKUN 
     text = re.sub('\u064b', '', text) #ARABIC FATHATAN 
     text = re.sub('\u064e', '', text) #ARABIC FATHA 
     text = re.sub('\u0650', '', text) #ARABIC KASRA 
     text = re.sub('\xa0',  ' ', text) #BAD SPACES 
     return text

data['review'] = data['review'].apply(lambda x: remove_shadda(x))

data['review'] = data['review'].apply(lambda x: remove_punct(x))
data['review'] = data['review'].str.replace('[{}]'.format(string.punctuation), '')
clean = re.compile('<.*?>')

regex = re.compile('[a-zA-Z]')
data['review'] = data['review'].map(lambda x: re.sub('[a-zA-Z]', '', x))
data['review'] = data['review'].map(lambda x: re.sub(r'&',' ', x))
data['review'] = data['review'].map(lambda x: re.sub('<[^>]+>', '', x))
data['review'] = data['review'].map(lambda x: re.sub(' href\s*=\s*\"[^\"]*', '', x))
data['review'] = data['review'].map(lambda x: re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', x)) # remove URLs
data['review'] = data['review'].map(lambda x: re.sub('@[^\s]+', 'AT_USER', x))
data['review'] = data['review'].map(lambda x: re.sub(r'#([^\s]+)', r'\1', x))
data['review'] = data['review'].apply(lambda x: remove_punct(x))
data['review'] = data['review'].str.replace('[{}]'.format(string.punctuation), '')
clean = re.compile('<.*?>')
data['review'] = data['review'].map(lambda x: re.sub(clean,'', x))
data['review'] = data['review'].map(lambda x: re.sub('\u200c',' ', x))

data['review'] = data['review'].apply(lambda x: remove_space(x))

data['review'] = data['review'].apply(lambda x: emoji(x))

data['review']

nan_value = float("NaN")
data.replace(" ", nan_value, inplace=True)
data.dropna(subset = ["review"], inplace=True)

data['review']

data = data.drop_duplicates(subset=['review'], keep='first')
data = data.reset_index(drop=True)
data

data['comment_len_by_words'] = data['review'].apply(lambda t: len(hazm.word_tokenize(t)))

min_max_len = data["comment_len_by_words"].min(), data["comment_len_by_words"].max()
print(f'Min: {min_max_len[0]} \tMax: {min_max_len[1]}')

def data_gl_than(data, less_than=100.0, greater_than=0.0, col='comment_len_by_words'):
    data_length = data[col].values

    data_glt = sum([1 for length in data_length if greater_than < length <= less_than])

    data_glt_rate = (data_glt / len(data_length)) * 100

    print(f'Texts with word length of greater than {greater_than} and less than {less_than} includes {data_glt_rate:.2f}% of the whole!')

data_gl_than(data, 256, 3)

minlim, maxlim = 3, 2110

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def cleaning(text):
    text = text.strip()
   
    # cleaning htmls
    text = cleanhtml(text)
    
    # normalizing
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    
    return text

# cleaning comments
data['cleaned_comment'] = data['review'].apply(cleaning)


# calculate the length of comments based on their words
data['cleaned_comment_len_by_words'] = data['cleaned_comment'].apply(lambda t: len(hazm.word_tokenize(t)))

# remove comments with the length of fewer than three words
data['cleaned_comment_len_by_words'] = data['cleaned_comment_len_by_words'].apply(lambda len_t: len_t if minlim < len_t <= maxlim else len_t)
data = data.dropna(subset=['cleaned_comment_len_by_words'])
data = data.reset_index(drop=True)
data = data.drop_duplicates(subset=['cleaned_comment'], keep='first')

print(data.head())

data = data[['cleaned_comment', 'sentiment']]
data.columns = ['review', 'sentiment']


def clean(text):
    intab='۱۲۳۴۵۶۷۸۹۰١٢٣٤٥٦٧٨٩٠'
    outtab='12345678901234567890'
    translation_table = str.maketrans(intab, outtab)
    text = text.translate(translation_table)
    tex = re.sub(r"(?:\@|https?\://)\s+"," ",text)
    text = re.findall(r"[A-Za-z._]+|[^A-Za-z\W]+",text,re.UNICODE)
    text = ' '.join(word for word in text)
    cleanr = re.compile('<.*?>')
    cleanr = re.compile(r'<[^>]+>')
    text = re.sub(cleanr,'',text)
    text = re.sub(r"""\d""",'',text)
    text = re.sub('\r?\n','.',text)
    text = re.sub(r"-{3}",'',text)
    text = re.sub(r"-{2}",'',text)
    text = re.sub(r"""\s*\.{3,}""",u'.',text)
    text = re.sub(r"""\s*\.{2,}""",u'.',text)
    text = re.sub(r"""\s+(ن؟می)\s+""",r'\1',text)
    text = re.sub(r"""(!){2,}""",r'\1',text)
    text = re.sub(r"""(/ ){2,}""",'',text)
    text = re.sub(r"""( /){2,}""",'',text)
    text = re.sub(r"""(//){2,}""",'',text)
    text = re.sub(r"""(/){2,}""",'',text)
    text = re.sub(r"""(؟){2,}""",r'\1',text)
    text = re.sub(r"""_+""","",text)
    text = re.sub(r"""[ ]+""",r' ',text)
    text = re.sub(r"""([\n]+)[\t]*""",r'\1',text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    p = re.compile(r'<.*?>')
    #text = re.findall(r"[^A-Za-z\W]+",text,re.UNICODE)
    #text = re.sub(r"""(/){1,}""",'-',text)
    #text = re.findall(r"[^\dA-Za-z\W]+|\d+",text,re.UNICODE)
    #text = ' '.join(word for word in text)
    #text = p.sub('', text)
    #text = re.sub(r"""\\d{2-3}\s+[^\dA-Za-z\W]\s+\\d{2-3}""",r'\1',text)
    #text = re.sub(r"""\s+(ی)ها|?(?(ن))""",r'\1',text)
    text = re.sub(r"""product""","",text)
    text = re.sub(r"""dkp""","",text)
    text = re.sub(r"""br""","",text)
    text = re.sub(r"""mm""","",text)
    text = text.strip()

    return text

data['review'] = data['review'].apply(lambda x: clean(x))

data['sentiment'] = data['sentiment'].fillna(-1)



data.to_csv("test data/phase 1/test1.csv", index = False)