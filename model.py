# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:17:43 2020

@author: Douches
"""
import numpy as np
import pickle
import operator
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import sequence
# load the data
context = np.load('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/context_indexes.npy',allow_pickle=True)
final_target = np.load('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/target_indexes.npy',allow_pickle=True)
with open('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)


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
#temp soln: reducing the size of data to be processed
final_target_1=final_target[:100]
context_1 = context[:100]

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

# initialize and populate the outputs to the Keras model. The output is the same as the target, but shifted one time step to the left
# (teacher forcing)

outs = np.zeros((context_1.shape[0], maxLen, vocab_size))


for pos,i in enumerate(final_target_1):
	for pos1,j in enumerate(i):
		if pos1 > 0:
			outs[pos, pos1-1, j] = 1
	if pos%1000 == 0: print ('{} entries completed'.format(pos))
# pad the sequences so that they can be fed into the embedding layer
final_target_1 = sequence.pad_sequences(final_target_1, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')
context_1 = sequence.pad_sequences(context_1, maxlen = 20, dtype = 'int32', padding = 'post', truncating = 'post')
# load the pre-trained GloVe vectors into the embedding layer
embed_layer = Embedding(input_dim = vocab_size, output_dim = 50, trainable = True, )
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

# encoder and decoder global LSTM variables with 300 units
#LSTM_cell = LSTM(300, return_state = True)
#LSTM_decoder = LSTM(300, return_state = True, return_sequences = True)

# final dense layer that uses TimeDistributed wrapper to generate 'vocab_size' softmax outputs for each time step in the decoder lstm
input_context = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_context')
input_ctx_embed = embed_layer(input_context)
encoder_lstm, h1, c1 = LSTM(256, return_state = True, return_sequences = True)(input_ctx_embed)
_, h3, c3 = LSTM(256, return_state = True,return_sequences = True)(encoder_lstm)
encoder_states = [h1, c1,h3,c3]


input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_target')
input_tar_embed = embed_layer(input_target)
# the decoder lstm uses the final states from the encoder lstm as the initial state
decoder_lstm, context_h, context_c = LSTM(256, return_state = True, return_sequences = True)(input_tar_embed, initial_state = [h3,c3],)
final, context_h3, context_c3 = LSTM(256, return_state = True, return_sequences = True)(decoder_lstm)
dense_layer=Dense(vocab_size, activation = 'softmax')
output = TimeDistributed(dense_layer)(final)
output=Dropout(0.2)(output)
model = Model([input_context, input_target], output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit([context_1, final_target_1], outs, epochs = 100, batch_size = 256)


# import required packages
import numpy as np
import re


# for initial filtering
maxlen = 12
maxLen = 20

# import the dictionary
with open('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/dictionary.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

# import the reverse dictionary
with open('C:/Users/Kevin Spiceywhinner/Desktop/Pedro/reverse_dictionary.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

# the question asked to the chatbot
question = 'how are you doing'

# preprocessing to make the data into the format required by the model, same as during training
a = question.split()
for pos,i in enumerate(a):
    a[pos] = re.sub('[^a-zA-Z0-9 .,?!]', '', i)
    a[pos]= re.sub(' +', ' ', i)
    a[pos] = re.sub('([\w]+)([,;.?!#&\'\"-]+)([\w]+)?', r'\1 \2 \3', i)
    if len(i.split()) > maxlen:
            a[pos] = (' ').join(a[pos].split()[:maxlen])
            if '.' in a[pos]:
                ind = a[pos].index('.')
                a[pos] = a[pos][:ind+1]
            if '?' in a[pos]:
                ind = a[pos].index('?')
                a[pos] = a[pos][:ind+1]
            if '!' in a[pos]:
                ind = a[pos].index('!')
                a[pos] = a[pos][:ind+1]

question = ' '.join(a).split()

# make the question into an array of the corresponding indexes
question = np.array([word_to_index[w] for w in question])

# pad sequences
question = sequence.pad_sequences([question], maxlen = 20)

# Keras model used to train, so that we define the variables (tensors) that ultimately go into the infernce model
#input_context = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_context')
#input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'output_context')

#input_ctx_embed = embed_layer(input_context)
#input_tar_embed = embed_layer(input_target)

#encoder_lstm, context_h, context_c = LSTM(256, return_state = True, return_sequences = True)(input_ctx_embed)
#encoder_lstm1, context_h1, context_c1 = LSTM(256, return_state = True, return_sequences = True)(encoder_lstm)
#
#decoder_lstm, h, _ = LSTM(256, return_state = True, return_sequences = True)(input_tar_embed, initial_state = [context_h1, context_c1],)
#decoder_lstm1, h1, _ = LSTM(256, return_state = True, return_sequences = True)(decoder_lstm)

#output = Dense(vocab_size, activation = 'softmax')(decoder_lstm1)

# Define the model for the input (question). Returns the final state vectors of the encoder LSTM
context_model = Model(input_context, encoder_states)

# define the inputs for the decoder LSTM
target_h = Input(shape = (256, ))
target_c = Input(shape = (256, ))
target_h2 = Input(shape = (256, ))
target_c2 = Input(shape = (256, ))
decoder_states_inputs = [target_h, target_c,target_h2,target_c2]
# the decoder LSTM. Takes in the embedding of the initial word passed as input into the decoder model (the 'BOS' tag), 
# along with the final states of the encoder model, to output the corresponding sequences for 'BOS', and the new LSTM states.  
target, h, c = LSTM(256, return_state = True, return_sequences = True)(input_tar_embed, initial_state = decoder_states_inputs[0:2])
target2, h2, c2 = LSTM(256, return_state = True, return_sequences = True)(target, initial_state=decoder_states_inputs[-2:])
decoder_states=[h,c,h2,c2]
dec_output = Dense(vocab_size, activation = 'softmax')(target2)
target_model = Model(
[input_target] + decoder_states_inputs,
[dec_output] + decoder_states)
#target_model = Model([input_target, target_h, target_c], [output, h2, c2])
target_model.summary()
# pass in the question to the encoder LSTM, to get the final encoder states of the encoder LSTM
states_value= context_model.predict(question)

#reverse_input_char_index = dict(
#    (i, char) for char, i in index_to_word.items())
#reverse_target_char_index = dict(
#    (i, char) for char, i in index_to_word.items())
# i keeps track of the length of the generated answer. This won't allow the model to genrate sequences with more than 20 words.
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
# =============================================================================
# def decode_sequence(input_seq, encoder_model, decoder_model):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)
# 
#     # Generate empty target sequence of length 1.
#     target_seq = np.zeros((1, 1, maxLen))
#     # Populate the first character of target sequence with the start character.
#     target_seq[0, 0, index_to_word['BOS']] = 1.
# 
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = []  #Creating a list then using "".join() is usually much faster for string creation
#     while not stop_condition:
#         to_split = decoder_model.predict([target_seq] + states_value)
# 
#         output_tokens, states_value = to_split[0], to_split[1:]
# 
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, 0])
#         sampled_char = index_to_word[sampled_token_index]
#         decoded_sentence.append(sampled_char)
# 
#         # Exit condition: either hit max length
#         # or find stop character.
#         if sampled_char == 'EOS' or len(decoded_sentence) > 20:
#             stop_condition = True
# 
#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1, maxLen))
#         target_seq[0, 0, sampled_token_index] = 1.
# 
#     return "".join(decoded_sentence)
# =============================================================================


# =============================================================================
# for seq_index in range(20):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     input_seq = input_context[seq_index: seq_index + 1]
#     decoded_sentence = decode_sequence(question,context_model,target_model)
#     #print('-')
#    # print('Input sentence:', input_texts[seq_index])
#     #print('Target sentence:', target_texts[seq_index])
#     print('Decoded sentence:', decoded_sentence)
# # initialize the answer that will be generated for the 'BOS' input. Since we have used pre-padding for padding sequences,
# # the last token in the 'answer' variable is initialised with the index for 'BOS'.
# answer = np.zeros((1, maxLen))
# answer[0, -1] = word_to_index['BOS']
# =============================================================================

