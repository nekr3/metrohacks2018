
from __future__ import print_function
import os
import numpy as np
np.random.seed(1337) #re-seed generator

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
#imports from keras for neural net
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import collections, numpy, csv
import sys
import re

def loadGloveEmbeddings():
    #Load Glove, a model of words to numbers
    # Stores a dictionary of words, with numbers corresponding
    print('Indexing word vectors.')
    BASE_DIR = '/media/hdd0/unraiddisk1/student/newsgroup' #where glove file is
    GLOVE_DIR = BASE_DIR + '/'
    GLOVE_DIR = BASE_DIR + '/glove.6B/'#accesses glove file
    embeddings_index = {} #opens Glove
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]#sets the word to 0th value in array
        
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    #index mapping words in the embeddings set
    #to their embedding vector
    
    f.close()
    return embeddings_index

embeddings_index = loadGloveEmbeddings() #opens Glove

print('Found %s word vectors.' % len(embeddings_index))
# Loaded Glove.
#embeddings_index is a map. ex: 'cat' => array(100)

def loadtrain():
    data = []
    labels = []
    with open("merged2.csv") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for line in csvreader:
            id = line[11]
            review = line[6]
            if review != "body":
                sentiment = line[11]
                labels.append(1 if (sentiment == '1') else 2 if (sentiment == '2') else 0)
                data.append(review)
    y = to_categorical(labels)
    return (data,y)

(train,y) = loadtrain()

def loadtest():
    data = []
    ids = []
    with open("testData.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            id = line[0]
            if id != 'id':
                review = line[1]
                data.append(review)
                ids.append(id)
    return (data,ids)

#(text_text,test_ids) = (["Fuck liberal gun control and long live the second amendment!"],[[0,0,1]])
#(test_text,test_ids) = loadtest()

corpi = [train]#, test_text]

def create_embedding_matrix(EMBEDDING_DIM, MAX_NB_WORDS, word_index):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return (nb_words, embedding_matrix)

MAX_SEQUENCE_LENGTH = 1000

def create_tokenizer_and_embedding(MAX_SEQUENCE_LENGTH, train):
    MAX_NB_WORDS = 5000 #sets up for padding
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train)
    (nb_words, embedding_matrix) = create_embedding_matrix(EMBEDDING_DIM, MAX_NB_WORDS, tokenizer.word_index)
    # load pre-trained word embeddings into an Embedding layer
    # set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    return (tokenizer, embedding_layer)

(tokenizer, embedding_layer) = create_tokenizer_and_embedding(MAX_SEQUENCE_LENGTH, corpi[0])

def create_sequences(MAX_SEQUENCE_LENGTH, tokenizer, corpi):
    MAX_NB_WORDS = 5000 #sets up for padding
    EMBEDDING_DIM = 100
    padded_sequences = []
    for corpus in corpi:
        corpi_sequence = tokenizer.texts_to_sequences(corpus)
        padded_sequences.append(pad_sequences(corpi_sequence, maxlen=MAX_SEQUENCE_LENGTH))
    return padded_sequences

padded_sequences = create_sequences(MAX_SEQUENCE_LENGTH, tokenizer, corpi)

data = padded_sequences[0]

VALIDATION_SPLIT = 0.3 #splits in train and test
# train is 70%, test 30%

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = y[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

#sets train and test(data and labels)
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

lper = 0;
cper = 0;
rper = 0;
for x in y_val :
    if (x[0] == 1) :
        lper = lper + 1
    elif (x[1] == 1) :
        cper = cper + 1
    else :
        rper = rper + 1
print(str(lper) + " " + str(cper) + " " + str(rper))
#x_test = padded_sequences[1]

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(2)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=15, batch_size=128)
#from sklearn.metrics import confusion_matrix
#cnf_matrix = confusion_matrix(y_train, y_pred)
# >>> model.predict(x_test)
# predict instead of fit for small sample

model.save_weights("mymodel3.h5")
model_json = model.to_json()
with open("mymodel3.json", "w") as json_file:
    json_file.write(model_json)

import pickle
pickle.dump( tokenizer, open( "tokenizer2.pickle", "wb" ) )

#test_sequences = create_sequences(MAX_SEQUENCE_LENGTH, tokenizer, [test_text])
#predictions = model.predict(test_sequences, batch_size= 256, verbose=0)

#Y_test_pred = np.argmax(predictions,axis=1)

# Save the predicitions in Kaggle format
#np.savetxt("predictions.csv", np.c_[test_ids,Y_test_pred], delimiter=',', header = 'id,sentiment', comments = '', fmt='%s')

