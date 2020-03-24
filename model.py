import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Activation, Flatten
import os
import numpy
import pandas

x = "fjdsiofd"#change to x values of data
y = "fjsiofjsiofsdj"#change to y values of data

#preprocess into NLP

train_x = x[:len(x)/2]
test_x = x[len(x)/2:]
train_y = y[:len(y)/2]
test_y = y[len(y)/2:]

model = Sequential()#model may need a bit of editing since there are wayy to many dense layers here
model.add(Dense(7))
model.add(Dense(256))\
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Dense(12))
model.add(Dense(3))
model.add(Activation('relu'))
model.compile(loss='categorical_crossentropy', ##optimizer=opt,
              metrics=['accuracy'])

#model.fit

#model.evaluate

#model.save_weights

