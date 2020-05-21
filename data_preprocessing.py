# -*- coding: utf-8 -*-
"""
Created on Fri May 15 02:40:43 2020

@author: Douches
"""

import numpy as np
import re
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
import time
from csv import reader
import pandas as pd
import collections
import nltk
import pickle

from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ds=[]
with open("C:/Users/Kevin Spiceywhinner/Desktop/TestDataset/BanksFAQ.csv", 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        ds.append(row)
questions=[]
answers=[]
for line in ds:
	questions.append(line[0])
	answers.append(line[1])
		

maxlen = 12
for pos,i in enumerate(answers):
    answers[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
    answers[pos] = re.sub(' +', ' ', i)
    answers[pos] = re.sub('([\w]+)([,;.?!#&-\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
    if len(i.split()) > maxlen:
        answers[pos] = (' ').join(answers[pos].split()[:maxlen])
        if '.' in answers[pos]: 
            ind = answers[pos].index('.')
            answers[pos] = answers[pos][:ind+1]
        if '?' in answers[pos]:
            ind = answers[pos].index('?')
            answers[pos] = answers[pos][:ind+1]
        if '!' in answers[pos]:
            ind = answers[pos].index('!')
            answers[pos] = answers[pos][:ind+1]

context = list(questions)
for pos,i in enumerate(context):
    context[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
    context[pos] = re.sub(' +', ' ', i)
    context[pos] = re.sub('([\w]+)([,;.?!#&\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
    if len(i.split()) > maxlen:
            context[pos] = (' ').join(context[pos].split()[:maxlen])
            if '.' in context[pos]:
                ind = context[pos].index('.')
                context[pos] = context[pos][:ind+1]
            if '?' in context[pos]:
                ind = context[pos].index('?')
                context[pos] = context[pos][:ind+1]
            if '!' in context[pos]:
                ind = context[pos].index('!')
                context[pos] = context[pos][:ind+1]


counts = {}
for words in context+answers:
    for word in words.split():
        counts[word] = counts.get(word,0) + 1
	

	

#dict_words['BOS']=len(dict_words)+1
#dict_words['EOS']=len(dict_words)+1
#dict_words['<OUT>']=len(dict_words)+1
word_to_index = {}
for pos,i in enumerate(counts.keys()):
    word_to_index[i] = pos

# reverse dictionary mapping indexes to words
index_to_word = {}
for k,v in word_to_index.items():
    index_to_word[v] = k

# apply the dictionary to the context and target data
final_target = np.array([[word_to_index[w] for w in i.split()] for i in answers])
context = np.array([[word_to_index[w] for w in i.split()] for i in context])	

	 
# save files
np.save('context_indexes', context)

np.save('target_indexes', final_target)

with open('dictionary.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)

with open('reverse_dictionary.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
	 

