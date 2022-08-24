import jaydebeapi
import pandas as pd
import numpy as np
import regex as re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
import konlpy
from konlpy.tag import Mecab, Okt, Kkma
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize_and_postag(df, content_column_name):
    """tokenize the news contents and pos-tagging (ex. NNG, NNP, VV, VA)

    Args:
        df (dataframe): dataframe for tokenize
        content_column_name (str): column name which have 'news content'

    Returns:
        tokenized (list): list with sentence token
        poss (list): list with pos tag for each token
    """
    tokenized = []
    poss = []
    mecab = Mecab()
    for i in range(len(df[content_column_name])):
        sentence = df[content_column_name][i]
        tk = mecab.morphs(sentence)
        tokenized.append(tk)
        pos = mecab.pos(sentence)
        poss.append(pos)
        
    return tokenized, poss

def clean_word(df, pos_tagged_column_name, sw):
    """get rid of token pos-tagging (ex. NNG, NNP, VV, VA)

    Args:
        df (dataframe): dataframe for tokenize
        content_column_name (str): column name which have 'news content'

    Returns:
        tokenized (list): list with sentence token
        poss (list): list with pos tag for each token
    """
    
    cleans = []
    for i in range(len(df[pos_tagged_column_name])):
        sentence = df[pos_tagged_column_name][i]
        clean_words = []
        for word, pos in sentence:
            if word not in sw and pos in ['NNG', 'NNP', 'VV', 'VA', 'VX', 'MM',  'XR']:
                clean_words.append(word)
        cleans.append(clean_words)
    return cleans

def clean_word2(df, pos_tagged_column_name):
    cleans = []
    for i in range(len(df[pos_tagged_column_name])):
        sentence = df[pos_tagged_column_name][i]
        clean_words = []
        for word, pos in sentence:
            if pos in ['NNG', 'NNP', 'VV', 'VA', 'VX', 'MM',  'XR']:
                clean_words.append(word)
        cleans.append(clean_words)
    return cleans
                
def to_ngrams(words, n):
    ngrams = []
    for b in range(0, len(words) - n + 1):
        ngrams.append(tuple(words[b: b+n]))
    return ngrams

def get_ngrams(cleans, n):
    ngrm = []
    for i in range(len(cleans)):
        ngr = []
        for ngrams in to_ngrams(cleans[i], n):
            ngr.append(ngrams)
        ngrm.append(ngr)

    return ngrm

def get_most_frequent(df, n_gram_column_name, n):
    lists = []
    for i in range(len(df)):
        lists.append(df[n_gram_column_name][i])
    flat_list = []
    for sublist in lists:
        for item in sublist:
            flat_list.append(item)

    return Counter(flat_list).most_common(n)

def get_doc_list(df, cleans):
    doc_list = []
    for i in range(len(df)):
        nouns = ' '.join(map(str, cleans[i]))
        doc_list.append(nouns)
    return doc_list

def get_tfidf_ngram(tfidf, df, cleans, doc_list, val):

    x = tfidf.fit(doc_list)
    tfidf_df = pd.DataFrame(x.transform(doc_list).toarray(), columns = sorted(tfidf.vocabulary_))
    important_words = []
    for i in range(0, len(tfidf_df)):
        a = []
        imp_word = tfidf_df.iloc[i, :].index[tfidf_df.iloc[i, :].values>val].tolist()
        important_words.append(imp_word)
    
    return important_words