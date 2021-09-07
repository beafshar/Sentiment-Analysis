

# !pip install -q hazm
# !pip install -q clean-text[gpl]
# !pip install plotly
# !pip install python-settings

from python_settings import settings
import pandas as pd
import hazm
import csv, re, pickle

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

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

# data_positive1 = pd.read_csv("phase1/positive1_phase1.csv")


# data_positive2 = pd.read_csv("phase1/positive2_phase1.csv")


# data_negative = pd.read_csv("phase1/negative_phase1.csv")


# data_neutral = pd.read_csv("phase1/neutral_phase1.csv")


# df = data_neutral.append([data_negative, data_positive1, data_positive2])

df = pd.read_csv("test data/phase 1/test1.csv")

# print data information
print('data information')
print(df.info(), '\n')

# print missing values information
print('missing values stats')
print(df.isnull().sum(), '\n')

def print_duplicate_by_column(df, col_name):
    ids = df[col_name]
    dup_df = df[ids.isin(ids[ids.duplicated()])].sort_values(col_name)
    pd.set_option('display.max_rows', df.shape[0]+1)
    return dup_df

print_duplicate_by_column(df,'review')

#remove duplicate rows
df = df.drop_duplicates(subset = None, keep='first')

#remove unvalid rows
df = df.drop_duplicates(subset='review')

print_duplicate_by_column(df,'review')

print('data information')
print(df.info(), '\n')

df.to_csv("test data/phase 2/final.csv", index = False)

df = pd.read_csv("test data/phase 2/final.csv")


# mj = 0

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def CleanPersianText(text):
    text = str(text)
    _normalizer = hazm.Normalizer()
    _lemmatizer = hazm.Lemmatizer()
    text = cleanhtml(text)
    text = _normalizer.normalize(text)
    text = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', text)
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)


    return text
    


df['review'] = df['review'].map(CleanPersianText)

df.to_csv("test data/phase 2/final_data2.csv", index = False)

#coment line below
df = pd.read_csv("test data/phase 2/final_data2.csv")
print(df['review'].head())

df['comment_len'] =df['review'].apply(lambda comment: len(hazm.word_tokenize(comment)))
min_max_len = df['comment_len'].min(), df['comment_len'].max()
print(f'Min: {min_max_len[0]} \tMax: {min_max_len[1]}')

df.sort_values(by='comment_len')

# remove comments with the length of fewer than 1 word 
df['comment_len'] = df['comment_len'].apply(lambda len_t: len_t if 0 < len_t  else None)
df = df.dropna(subset=['comment_len'])
df = df.reset_index(drop=True)

def show_comment_length_distribution(df):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['comment_len']
    ))

    fig.update_layout(
        title_text='Comment lenght distribution',
        xaxis_title_text='Word Count',
        yaxis_title_text='Frequency',
        bargap=0.2,
        bargroupgap=0.2)

    fig.show()
# show_comment_length_distribution(df)

def show_sentiment_distribution(df):
    fig = go.Figure()

    groupby_sentiment = df.groupby('sentiment')['sentiment'].count()

    fig.add_trace(go.Bar(
        x=list(sorted(groupby_sentiment.index)),
        y=groupby_sentiment.tolist(),
        text=groupby_sentiment.tolist(),
        textposition='auto'
    ))

    fig.update_layout(
        title_text='Sentiment distribution',
        xaxis_title_text='Sentiment',
        yaxis_title_text='Frequency',
        bargap=0.2,
        bargroupgap=0.2)

    fig.show()
# show_sentiment_distribution(df)

# train, test = train_test_split(df, test_size=0.1, random_state=1, stratify=df['sentiment'])
# train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train['sentiment'])

# train = train.reset_index(drop=True)
# valid = valid.reset_index(drop=True)
# test = test.reset_index(drop=True)

# print(train.shape)
# print(valid.shape)
# print(test.shape)

# train.to_csv('phase2/final/train.csv')
# test.to_csv('phase2/final/test.csv')
# valid.to_csv('phase2/final/valid.csv')

df.to_csv("test data/final_test.csv")