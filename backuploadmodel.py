from __future__ import print_function
from keras.models import model_from_json
json_file = open('mymodel3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mymodel3.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

from sklearn.metrics import confusion_matrix
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
    with open("data.csv") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for line in csvreader:
            id = line[1]
            review = line[0]
            if review != "body":
                sentiment = line[1]
                labels.append(1 if (sentiment == '1') else 2 if (sentiment == '2') else 0)
                data.append(review)
		print(review)
		print(sentiment)
    y = to_categorical(labels)
    return (data,y)

# Loaded Glove.
#(train,y) = loadtrain()



def loadxytext(filename): #loads csv where first column is type (y), second column is value (x)
    xvalues, yvalues, types = [], [], []
    with open(filename) as f:
        for row in csv.reader(f, delimiter=','): #line in f: #splits each line at comma
            itemtype,text = row[0], row[1]
            typenum = -1
            if itemtype in types:
                typenum = types.index(itemtype)
            else:
                typenum = len(types)
                types.append(itemtype)
            yvalues.append(typenum)
            xvalues.append(re.sub(r'[^a-zA-Z ]+','', text).lower())
    return (np.array(xvalues),np.array(yvalues),to_categorical(yvalues),types) #one-hot y

#(xvalues,yvalues,yvalues_categorical,types) = loadxytext("alldata")
(train,yvalues,yvalues_categorical,types) = loadxytext("merged2.csv")
for i in range(len(types)):
    print("Note that the type '%s' is mapped to %s" % (types[i] , i))

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

y_pred = loaded_model.predict(data, batch_size=128, verbose=0)
print(y_pred)

#score = loaded_model.evaluate(data, y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
