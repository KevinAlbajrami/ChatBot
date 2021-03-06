# -*- coding: utf-8 -*-
"""chatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xmpQlSuZyTCOkAOW4RMMv5tDrmoeql36

Firstly, importing the packages and reading the values from the dataset.
Note the path needs to be changed with the path where the dataset is located.
"""

import numpy as np
import re
import time
from csv import reader
import pandas as pd
import collections
import nltk
import pickle
import operator
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import utils
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

#importing the dataset
ds=[]
with open("--path to dataset--", 'r',encoding='utf-8') as read_obj:
	csv_reader = reader(read_obj)
	for row in csv_reader:
		ds.append(row)

"""Declaring variables"""

lineLength=20 # determine how long to be the line read from dataset
b_s=128 #batch size
DROPOUT=0.4
LAYER_SIZE=300
maxLen=20
valSize=400 #reserve first 400 rows for validation--to be changed to ~20% of the traning size
EPOCHS=200
#Define callback to monitro condition if training has to stop beore the epochs are fnished
CALL_BACKS = EarlyStopping(monitor='loss', patience=15)

"""In this dataset, questions are located in rows 1 and 2, and the answers are located in row 3. However we are ignoring row 2 and only keeping row 1 as a question. For questions/answers longer that maximum line length, the sentence will be shortened to the first lineLength words of the sentence."""

questions_v0=[]
answers_v0=[]
for line in ds:
	questions_v0.append(line[1])
	answers_v0.append(line[3])
#The dataset has many entries at this point so we are selecting only the first 3700
#this is done in order to reduce the testing size for faster calculations and because it takes up
#less ram
answers_v1=[]
questions_v1=[]
#Since our maxlength is 20, only the first 19 words of each question/answer are taken under consideration
str=" "
for line in answers_v0:
	if(len(line.split())>lineLength):
		mid=line.split()[:lineLength]
		answers_v1.append(str.join(mid))
	else:
		answers_v1.append(line)

for line in questions_v0:
	if(len(line.split())>lineLength):
		mid=line.split()[:lineLength]
		questions_v1.append(str.join(mid))
	else:
		questions_v1.append(line)

"""Due to the large amount of time it would take to process all the dataset, we are only taking the first 8000 entries."""

#answers_v1=answers_v1[:8000]
#questions_v1=questions_v1[:8000]

"""The following function sever to remove all the noise from the dataset including:

*   Symbols
*   Abbreviations
"""

#removing not important symbols and abbreviations
def clean_text(text):
	text=text.lower()
	text=re.sub(r"i'm","I am",text)
	text=re.sub(r"he's","he is",text)
	text=re.sub(r"she's","she is",text)
	text=re.sub(r"that's","that is",text)
	text=re.sub(r"what's","what is",text)
	text=re.sub(r"<br />","",text)
	text=re.sub(r"/n"," ",text)
	text=re.sub(r"\\n"," ",text)
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

#Applying the above function on each question/answer
questions=[]
answers=[]
for question in questions_v1:
	 question=clean_text(question)
	 questions.append(question)

for answer in answers_v1:
	 answer=clean_text(answer)
	 answers.append(answer)
 
maxlen = 20

"""Adding <BOS> and <EOS> tags to the answers to denote beginning of sentences and ending of sentences."""

final_target_v0 = ['BOS '+i+' EOS' for i in answers]
context_v0 = list(questions)

"""Removing Extra spaces '    ' -> ' '"""

final_target_v0 = list(pd.Series(final_target_v0).map(lambda x: re.sub(' +', ' ', x)))
context_v0 = list(pd.Series(questions).map(lambda x: re.sub(' +', ' ', x)))

"""Creating a counts list to store all the words in questions and answers. This list is later converted to a dictionary that maps every word to an index, and this one will finally be reversed so that every index points to its corresponding word"""

counts = {}
for words in context_v0+final_target_v0:
    for word in words.split():
        counts[word] = counts.get(word,0) + 1
#create a dictionary to associate each word with a specific index
word_to_index = {}
for pos,i in enumerate(counts.keys()):
	word_to_index[i] = pos

"""Converting answers and questions to integer sequences based on the word_to_index dictionary, delclared earlier"""

final_target = np.array([[word_to_index[w] for w in i.split()] for i in final_target_v0])
context = np.array([[word_to_index[w] for w in i.split()] for i in context_v0])

"""Here begins the seq2seq model part assignings new names to the lists as not to create confusion with the previous part.
final_target_1 refers to the answers
context_1 refers to the questions
"""

final_target_1 = final_target
context_1 = context

"""Before the training the data will be padded so that every sentence has the same length. The padding process is done by adding zeros to the end of every sentence. However at this point we have some words that are mapped to zeros in our lists and dictionaries, and by incrementing the indexes of each word by 1 this problem is solved."""

# shift the indexes of the context and target arrays too
for i,j in word_to_index.items():
    word_to_index[i] = j+1
# reverse dictionary
index_to_word = {}
for k,v in word_to_index.items():
    index_to_word[v] = k

for i in final_target_1:
    for pos,j in enumerate(i): i[pos] = j + 1
for i in context_1:
    for pos,j in enumerate(i): i[pos] = j + 1

"""Read the embedded 50 dimenssional GloVe file.
Note: replace the path with where the GloVe file is located
"""

# read in the 50 dimensional GloVe embeddings
def read_glove_vecs(file):
    with open(file, 'r',encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

words, word_to_vec_map = read_glove_vecs('/content/drive/My Drive/glove.6B.50d.txt')

"""Create an embedding matrix and add all the vecorized words of our vocabulary to it."""

vocab_size = len(word_to_index) + 1
#setting validation size to be used for the validation set
valSize=int(0.2*vocab_size)
# initialize the embedding matrix that will be used (50 is the GloVe vector dimension)
embedding_matrix = np.zeros((vocab_size, 50))
for word,index in word_to_index.items():
    try:
        embedding_matrix[index, :] = word_to_vec_map[word.lower()]
    except: continue

"""Generator method to produce model inputs and outputs. Among all teh questions and answers only bacth_size number of each are processed at a time. Every time the generator is run batch_size questions and answers are produced for the traning process. These sentences are padded, and then the answers are moved one step forward and hot coded as the output of the decoder."""

def generator(questions,answers, batch_size=32):
	num_samples = len(questions)
	while True: # Loop forever so the generator never terminates
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
		for offset in range(valSize, num_samples, batch_size):
			# Get the samples you'll use in this batch
			#batch_samples = samples[offset:offset+batch_size]
			question_samples = questions[offset:offset+batch_size]
			answer_samples = answers[offset:offset+batch_size]
			# Initialise X_train and y_train arrays for this batch
			ques_train = []
			ans_train = []
			for i in question_samples:
				ques_train.append(i)
			for i in answer_samples:
				ans_train.append(i)
            # Make sure they're numpy arrays (as opposed to lists)
			ques_train = np.array(ques_train)
			ans_train = np.array(ans_train)
			ans_pad = sequence.pad_sequences(ans_train, maxlen = maxLen, dtype = 'int32', padding = 'post', truncating = 'post')
			ques_pad = sequence.pad_sequences(ques_train, maxlen = maxLen, dtype = 'int32', padding = 'post', truncating = 'post')
			#print(ans_train)
			#print('ques',ques_pad)
			encoder_input_data = np.array( ques_pad )
			decoder_input_data = np.array( ans_pad )
			for i in range(len(ans_train)) :
				ans_train[i] = ans_train[i][1:]
			padded_answers = sequence.pad_sequences( ans_train , maxlen=maxLen , padding='post' )
			onehot_answers = utils.to_categorical( padded_answers , vocab_size )
			decoder_output_data = np.array( onehot_answers )		
			yield(encoder_input_data,decoder_input_data),decoder_output_data

"""Similar decoder to the above but to produce validation results, this decoder is only run once."""

def val_generator(questions,answers):
  num_samples = len(questions)
  question_samples = questions[:valSize]
  answer_samples = answers[:valSize]
  ques_train = []
  ans_train = []
  for i in question_samples:
    ques_train.append(i)
  for i in answer_samples:
    ans_train.append(i)
  ans_pad = sequence.pad_sequences(ans_train, maxlen = maxLen, dtype = 'int32', padding = 'post')
  ques_pad = sequence.pad_sequences(ques_train, maxlen = maxLen, dtype = 'int32', padding = 'post')
  encoder_input_data = np.array( ques_pad )
  decoder_input_data = np.array( ans_pad )
  for i in range(len(ans_train)) :
    ans_train[i] = ans_train[i][1:]
  padded_answers = sequence.pad_sequences( ans_train , maxlen=maxLen , padding='post' )
  onehot_answers = utils.to_categorical( padded_answers , vocab_size )
  decoder_output_data = np.array( onehot_answers )
  return(encoder_input_data,decoder_input_data),decoder_output_data

"""Defining input layers

*   input_context for the encoder
*   input_target for the decoder
"""

input_context = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_context')
input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_target')

"""Training model, includees and embeding layer. Each LSTM layer has size 300, there are three LSTM layers. each layer is composed of an encoder and adecoder where the encoder is fed data and passes it to its corresponing decoder. Finally dense layer, and dropout are applied."""

embed_layer = Embedding(input_dim = vocab_size, output_dim = 50, trainable = True,mask_zero=True )
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix],)
input_ctx_embed = embed_layer(input_context)
encoder_lstm,h1,c1 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(input_ctx_embed)
encoder_lstm2,h2,c2 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(encoder_lstm)
encoder_lstm2,h3,c3 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(encoder_lstm2)
encoder_states=[h1,c1,h3,c3]
input_tar_embed = embed_layer(input_target)
final1, context_h1, context_c1 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(input_tar_embed, initial_state = [h1,c1])
final2, context_h2, context_c2 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(final1, initial_state = [h2,c2])
final3, context_h3, context_c3 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(final2, initial_state = [h3,c3])
dense_layer=Dense(vocab_size, activation = 'softmax')
output = TimeDistributed(dense_layer)(final3)
output=Dropout(DROPOUT)(output)
model = Model([input_context, input_target], output)
Adam_1 = optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer = Adam_1, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

"""Train the model with the generated data, validation contains the validation data to be used. Epochs number, and steps for epoch since we are using generatora are also specified. Callbacks are defined in the top."""

train_generator=generator(context_1,final_target_1,batch_size=b_s)
spe = context_1.shape[0]/b_s
if context_1.shape[0] % b_s:
    spe += 1
validation=val_generator(context_1,final_target_1)
model.fit(train_generator,validation_data=validation ,epochs = 400,steps_per_epoch=spe,callbacks=[CALL_BACKS])

"""Define encoder model and decoder model from the main model, these will be part of the inference model."""

context_model = Model(input_context, encoder_states)
#define the inputs for the decoder LSTM
target_h = Input(shape = (LAYER_SIZE, ))
target_c = Input(shape = (LAYER_SIZE, ))
target_h1 = Input(shape = (LAYER_SIZE, ))
target_c1 = Input(shape = (LAYER_SIZE, ))
target_h3 = Input(shape = (LAYER_SIZE, ))
target_c3 = Input(shape = (LAYER_SIZE, ))
decoder_states_inputs = [target_h, target_c,target_h1,target_c1,target_h3,target_c3]
# the decoder LSTM takes in the embedding of the initial word passed as input into the decoder model (the 'BOS' tag) 
# along with the final states of the encoder model, to output the corresponding sequences for 'BOS', and the new LSTM states.  
target, h, c = LSTM(input_tar_embed, initial_state = decoder_states_inputs[:2])
target2, h2, c2 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(target, initial_state = decoder_states_inputs[2:4])
target2, h3, c3 = LSTM(LAYER_SIZE, return_state = True, return_sequences = True)(target2, initial_state = decoder_states_inputs[4:6])

decoder_states=[h,c,h2,c2,h3,c3]
dec_output = Dense(vocab_size, activation = 'softmax')(target)
target_model = Model(
[input_target] + decoder_states_inputs,
[dec_output] + decoder_states)

"""str_to_tokens is used to convert the question to a sequence of integers."""

# pass in the question to the encoder LSTM, to get the final encoder states of the encoder LSTM
def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(word_to_index[ word ] ) 
    return sequence.pad_sequences( [tokens_list] , maxlen=20 , padding='post')
# run the inference model
question='where is it'
#question=clean_text(question)

"""Talk to the chatbot, the question is tokenized and then the predicition is applied. The data goes through the context and decoder models and is then produced. 


```
for _ in range(1):
```
defines how many questions you want to ask in a row. in this case use 

```
Input('Enter a question instead of a single question')
```
"""

for _ in range(1):
	states_values = context_model.predict( str_to_tokens(question) )
	target_seq = np.zeros( ( 1 , 1) )
	target_seq[0, 0] = word_to_index['BOS']
	stop_condition = False
	result = ''
	while not stop_condition :
		dec_outputs , ha , ca, hb, cb,hc,cc= target_model.predict([ target_seq ] + states_values )
		sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
		sampled_word = None
		for word , index in word_to_index.items() :
			if sampled_word_index == index :
				result += ' {}'.format( word )
				sampled_word = word

		if sampled_word == 'EOS' or len(result.split()) > 10:
			stop_condition = True
		target_seq = np.zeros( ( 1 , 1 ) )
		target_seq[ 0 , 0 ] = sampled_word_index
		states_values = [ ha , ca, hb, cb, hc, cc]
	print( result )