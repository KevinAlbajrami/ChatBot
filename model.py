# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:17:43 2020

@author: Douches
"""
import numpy as np
import pickle
import operator
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import sequence
# load the data
context = np.load('C:/Users/Kevin Spiceywhinner/Desktop/context_indexes.npy',allow_pickle=True)
final_target = np.load('C:/Users/Kevin Spiceywhinner/Desktop/target_indexes.npy',allow_pickle=True)
with open('C:/Users/Kevin Spiceywhinner/Desktop/dictionary.pkl', 'rb') as f:
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
final_target_1=final_target[:50]
context_1 = context[:50]

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
			outs[pos, pos1 - 1, j] = 1
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
encoder_lstm, h1, c1 = LSTM(300, return_state = True, return_sequences = True)(input_ctx_embed)
encoder_lstm2,h2, c2 = LSTM(300, return_state = True, return_sequences = True)(encoder_lstm)
_,h3, c3 = LSTM(300, return_state = True)(encoder_lstm2)
encoder_states = [h1, c1, h2, c2,h3,c3]


input_target = Input(shape = (maxLen, ), dtype = 'int32', name = 'input_target')
input_tar_embed = embed_layer(input_target)
# the decoder lstm uses the final states from the encoder lstm as the initial state
decoder_lstm, context_h, context_c = LSTM(300, return_state = True, return_sequences = True)(input_tar_embed, initial_state = [h1, c1],)
decoder_lstm2, context_h2, context_c2 = LSTM(300, return_state = True, return_sequences = True)(decoder_lstm, initial_state = [h2, c2],)
final, context_h3, context_c3 = LSTM(300, return_state = True, return_sequences = True)(decoder_lstm2, initial_state = [h3, c3],)
dense_layer=Dense(vocab_size, activation = 'softmax')
output = TimeDistributed(dense_layer)(final)
output=Dropout(0.4)(output)
model = Model([input_context, input_target], output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit([context_1, final_target_1], outs, epochs = 50, batch_size = 32)