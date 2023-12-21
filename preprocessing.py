import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm
import json

def read_csv(fp):
    df = pd.read_csv(fp,  delimiter=',')
    return df

def preprocess(df):
    mbti_types = ['intj', 'intp', 'entj', 'entp',
                    'infj', 'infp', 'enfj', 'enfp',
                    'istj', 'isfj', 'estj', 'esfj',
                    'istp', 'isfp', 'estp', 'esfp']

    df['posts'] = df['posts'].replace(to_replace=r'http\S+', value='', regex=True)
    df['posts'] = df['posts'].str.replace('|||', ' ')
    clean_posts = []

    # print(df.posts[0])
    print("-------------------------")
    lemmatizer = WordNetLemmatizer()
    for p in tqdm(df.posts, desc="Cleaning Posts", unit="post"):
        text = p.lower()
        #remove emojis
        text = re.sub(':[a-zA-Z]{1,5}:','',text)
        text = word_tokenize(text)
        text = [w for w in text if w not in stopwords.words('english')] 
        text = [w for w in text if w not in mbti_types]
        text = [lemmatizer.lemmatize(w) for w in text]
        text = [w for w in text if (w != '...') and (w not in string.punctuation)]
        clean_posts.append(text)

    return clean_posts


df = read_csv('mbti_1.csv')
df.posts = preprocess(df)


output_fp = 'cleaned_posts.json'
df.to_json(output_fp, orient='records', lines=True)

