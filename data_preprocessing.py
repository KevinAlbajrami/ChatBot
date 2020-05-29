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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ds=[]
with open("C:/Users/Kevin Spiceywhinner/Desktop/TestDataset/dataTrain.csv", 'r',encoding='utf-8') as read_obj:
	csv_reader = reader(read_obj)
	for row in csv_reader:
		ds.append(row)
		
			
# =============================================================================
# for line in ds:
#  	if (line[2]==''):
#  		line[2]=line[1]
# =============================================================================
questions_v0=[]
answers_v0=[]
for line in ds:
	questions_v0.append(line[1])
	answers_v0.append(line[3])


answers_v0=answers_v0[:3700]
questions_v0 = questions_v0[:3700]
answers_v1=[]
questions_v1=[]
str=" "
for line in answers_v0:
	if(len(line.split())>19):
		mid=line.split()[:19]
		answers_v1.append(str.join(mid))
	else:
		answers_v1.append(line)

for line in questions_v0:
	if(len(line.split())>19):
		mid=line.split()[:19]
		questions_v1.append(str.join(mid))
	else:
		questions_v1.append(line)

def clean_text(text):
	text=text.lower()
	text=re.sub(r"i'm","I am",text)
	text=re.sub(r"he's","he is",text)
	text=re.sub(r"she's","she is",text)
	text=re.sub(r"that's","that is",text)
	text=re.sub(r"what's","what is",text)
	text=re.sub(r"<br />","",text)
	text=re.sub(r"/n"," ",text)
	text=re.sub(r"\n"," ",text)
	text=re.sub(r"\'ll"," will",text)
	text=re.sub(r"\'ve"," have",text)
	text=re.sub(r"\'re"," are",text)
	text=re.sub(r"\'d"," would",text)
	text=re.sub(r"won't","will not",text)
	text=re.sub(r"can't","cannot",text)
	text=re.sub(r"https"," ",text)
	text=re.sub(r"www."," ",text)
	text=re.sub(r".com"," ",text)
	text=re.sub(r"ÿ"," ",text)
	text=re.sub(r"[-()\"#/@;:<>{}+-=.?,|]"," ",text)
	return text


questions=[]
answers=[]
for question in questions_v1:
	 question=clean_text(question)
	 questions.append(question)

for answer in answers_v1:
	 answer=clean_text(answer)
	 answers.append(answer)
	 
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

# add Beginning of Sentence (BOS) and End of Sentence (EOS) tags to the 'target' data
final_target_v0 = ['BOS '+i+' EOS' for i in answers]

final_target_v0 = list(pd.Series(final_target_v0).map(lambda x: re.sub(' +', ' ', x)))
context = list(pd.Series(context).map(lambda x: re.sub(' +', ' ', x)))

counts = {}
for words in context+final_target_v0:
    for word in words.split():
        counts[word] = counts.get(word,0) + 1
	

#dict_words['BOS']=len(dict_words)+1
#dict_words['EOS']=len(dict_words)+1
#dict_words['<OUT>']=len(dict_words)+1
word_to_index = {}
for pos,i in enumerate(counts.keys()):
	word_to_index[i] = pos

# reverse dictionary mapping indexes to words
index_to_word = {}s
for k,v in word_to_index.items():
    index_to_word[v] = k


# apply the dictionary to the context and target data
final_target = np.array([[word_to_index[w] for w in i.split()] for i in final_target_v0])
context = np.array([[word_to_index[w] for w in i.split()] for i in context])
print(final_target[4])


 
# save files
np.save('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/context_indexes', context)

np.save('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/target_indexes', final_target)

with open('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/dictionary.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)

with open('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/reverse_dictionary.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
	 

