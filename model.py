# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:17:43 2020

@author: Douches
"""
import os
import numpy as np
import pickle
import operator
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import utils
# import required packages
import numpy as np
import re
from tensorflow.keras.models import model_from_json
# load the data
context = np.load('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/16000/context_indexes.npy',allow_pickle=True)
final_target = np.load('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/16000/target_indexes.npy',allow_pickle=True)
with open('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/16000/dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

weights_file = 'C:/Users/Kevin Spiceywhinner/Desktop/Pedro/my_model_weights20.h5'
# the indexes of the words start with 0. But when the sequences are padded later on, they too will be zeros.
# so, shift all the index values one position to the right, so that 0 is spared, and used only to pad the sequences
for i,j in word_to_index.items():
    word_to_index[i] = j+1

# reverse dictionary
index_to_word = {}
for k,v in word_to_index.items():
    index_to_word[v] = k

final_target_1 = final_target
context_1 = context

maxLen = 20
# shift the indexes of the context and target arrays too
for i in final_target_1:
    for pos,j in enumerate(i): i[pos] = j + 1
for i in context_1:
    for pos,j in enumerate(i): i[pos] = j + 1
	
	
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

words, word_to_vec_map = read_glove_vecs('C:/Users/Kevin Spiceywhinner/Desktop/TestDataset/glove.twitter.27B.50d.txt')

# since the indexes start from 1 and not 0, we add 1 to the no. of total words to get the vocabulary size (while initializing 
# and populating arrays later on, this will be required)
vocab_size = len(word_to_index) + 1

# initialize the embedding matrix that will be used (50 is the GloVe vector dimension)
embedding_matrix = np.zeros((vocab_size, 50))
for word,index in word_to_index.items():
    try:
        embedding_matrix[index, :] = word_to_vec_map[word.lower()]
    except: continue


embed_layer = Embedding(input_dim = vocab_size, output_dim = 50, trainable = True, )
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])


# final dense layer that uses TimeDistributed wrapper to generate 'vocab_size' softmax outputs for each time step in the decoder lstm
input_context = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_context')
input_ctx_embed = embed_layer(input_context)
encoder_lstm, h1, c1 = LSTM(256, return_state = True, return_sequences = True)(input_ctx_embed)
encoder_states = [h1, c1]
input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_target')
input_tar_embed = embed_layer(input_target)
final, context_h3, context_c3 = LSTM(256, return_state = True, return_sequences = True)(input_tar_embed, initial_state = [h1,c1],)
dense_layer=Dense(vocab_size, activation = 'softmax')
output = TimeDistributed(dense_layer)(final)
output=Dropout(0.35)(output)
model = Model([input_context, input_target], output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

def generator(questions,answers, batch_size=32,model=model):
	num_samples = len(questions)
	while True: # Loop forever so the generator never terminates
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
		for offset in range(0, num_samples, batch_size):
			# Get the samples you'll use in this batch
			#batch_samples = samples[offset:offset+batch_size]
			question_samples = questions[offset:offset+batch_size]
			answer_samples = answers[offset:offset+batch_size]
			# Initialise X_train and y_train arrays for this batch
			ques_train = []
			ans_train = []
			for i in question_samples:
				ques_train.append(i)
			for i in question_samples:
				ans_train.append(i)
            # Make sure they're numpy arrays (as opposed to lists)
			ques_train = np.array(ques_train)
			ans_train = np.array(ans_train)
			ans_pad = sequence.pad_sequences(ans_train, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')
			ques_pad = sequence.pad_sequences(ques_train, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')
			encoder_input_data = np.array( ques_pad )
			decoder_input_data = np.array( ans_pad )
			for i in range(len(ans_train)) :
				ans_train[i] = ans_train[i][1:]
			padded_answers = sequence.pad_sequences( ans_train , maxlen=20 , padding='post' )
			onehot_answers = utils.to_categorical( padded_answers , vocab_size )
			decoder_output_data = np.array( onehot_answers )		
			yield(encoder_input_data,decoder_input_data),decoder_output_data
			
#ignore this part of the code	
#with open('/content/model.json', 'r') as json_file:
#    json_savedModel= json_file.read()
#model=model_from_json(json_savedModel)
#model.load_weights("/content/model.h5")
#print('model loaded successfully')

batch_size=128
train_generator=generator(context_1,final_target_1,batch_size=batch_size)
spe = context_1.shape[0]/batch_size
if context_1.shape[0] % batch_size:
    spe += 1

model.fit(train_generator, epochs = 250,steps_per_epoch=spe)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# for initial filtering
maxlen = 12
maxLen = 20

# the question asked to the chatbot
question = 'how are you'

# preprocessing to make the data into the format required by the model, same as during training
# =============================================================================
# a = question.split()
# for pos,i in enumerate(a):
#     a[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
#     a[pos]= re.sub(' +', ' ', i)
#     a[pos] = re.sub('([\w]+)([,;.?!#&\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
#     if len(i.split()) > maxlen:
#             a[pos] = (' ').join(a[pos].split()[:maxlen])
#             if '.' in a[pos]:
#                 ind = a[pos].index('.')
#                 a[pos] = a[pos][:ind+1]
#             if '?' in a[pos]:
#                 ind = a[pos].index('?')
#                 a[pos] = a[pos][:ind+1]
#             if '!' in a[pos]:
#                 ind = a[pos].index('!')
#                 a[pos] = a[pos][:ind+1]
# 
# question = ' '.join(a).split()
# =============================================================================

# make the question into an array of the corresponding indexes
question = np.array([word_to_index[w] for w in question])
# pad sequences
question = sequence.pad_sequences([question], maxlen = 20)


context_model = Model(input_context, encoder_states)
# define the inputs for the decoder LSTM
target_h = Input(shape = (256, ))
target_c = Input(shape = (256, ))
decoder_states_inputs = [target_h, target_c]
# the decoder LSTM. Takes in the embedding of the initial word passed as input into the decoder model (the 'BOS' tag), 
# along with the final states of the encoder model, to output the corresponding sequences for 'BOS', and the new LSTM states.  
target, h, c = LSTM(256, return_state = True, return_sequences = True)(input_tar_embed, initial_state = decoder_states_inputs)
#target2, h2, c2 = LSTM(256, return_state = True, return_sequences = True)(target, initial_state=decoder_states_inputs[-2:])
decoder_states=[h,c]
dec_output = Dense(vocab_size, activation = 'softmax')(target)
target_model = Model(
[input_target] + decoder_states_inputs,
[target] + decoder_states)
#target_model = Model([input_target, target_h, target_c], [output, h2, c2])
target_model.summary()
# pass in the question to the encoder LSTM, to get the final encoder states of the encoder LSTM
states_value= context_model.predict(question)
answer = np.zeros((1, maxLen))
answer[0, -1] = word_to_index['BOS']
i = 1
# make a new list to store the words generated at each time step
answer_1 = []

# flag to stop the model when 'EOS' tag is generated or when 20 time steps have passed.
flag = 0

# run the inference model
while flag != 1:
    # make predictions for the given input token and encoder states
    prediction, h, c, h1, c1 = target_model.predict(
            [answer] + states_value)
    # from the generated predictions of shape (num_examples, maxLen, vocab_size), find the token with max probability
    token_arg = np.argmax(prediction[0, -1, :])
    
    # append the corresponding word of the index to the answer_1 list
    answer_1.append(index_to_word[token_arg])
    
    # set flag to 1 if 'EOS' token is generated or 20 time steps have passed
    if token_arg == word_to_index['EOS'] or i > 20:
        flag = 1
    # re-initialise the answer variable, and set the last token to the output of the current time step. This is then passed
    # as input to the next time step, along with the LSTM states of the current time step
    answer = np.zeros((1,maxLen))
    answer[0, -1] = token_arg
    states_value = [h, c, h1, c1]
    
    # increment the count of the loop
    i+=1
    
 # print the answer generated for the given question
print (' '.join(answer_1))
