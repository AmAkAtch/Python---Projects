import sys
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import os
import ast
from dotenv import load_dotenv

load_dotenv()

#open the plain text file in read mode and save contents in raw_text
with open("AI/shakespear.txt", "r", encoding='utf-8') as file:
    raw_text = file.read()

#keep entire text document in lowercase for ease
raw_text = raw_text.lower()


#defining the window size and empty lists which will be used to prepare model later
seq_length = 100
dataX = []
dataY = []

#Create uniq list of all charcters used in text and than create dictionaries for accessing char by index and index by characters
chars = sorted(list(set(raw_text)))
char_to_ind = {'\n': 0, ' ': 1, '!': 2, '#': 3, '$': 4, '%': 5, '&': 6, '(': 7, ')': 8, '*': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '?': 26, '[': 27, ']': 28, '_': 29, 'a': 30, 'b': 31, 'c': 32, 'd': 33, 'e': 34, 'f': 35, 'g': 36, 'h': 37, 'i': 38, 'j': 39, 'k': 40, 'l': 41, 'm': 42, 'n': 43, 'o': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 't': 49, 'u': 50, 'v': 51, 'w': 52, 'x': 53, 'y': 54, 'z': 55, 'æ': 56, '—': 57, '‘': 58, '’': 59, '“': 60, '”': 61, '•': 62, '™': 63}

int_to_char = {0: '\n', 1: ' ', 2: '!', 3: '#', 4: '$', 5: '%', 6: '&', 7: '(', 8: ')', 9: '*', 10: ',', 11: '-', 12: '.', 13: '/', 14: '0', 15: '1', 16: '2', 17: '3', 18: '4', 19: '5', 20: '6', 21: '7', 22: '8', 23: '9', 24: ':', 25: ';', 26: '?', 27: '[', 28: ']', 29: '_', 30: 'a', 31: 'b', 32: 'c', 33: 'd', 34: 'e', 35: 'f', 36: 'g', 37: 'h', 38: 'i', 39: 'j', 40: 'k', 41: 'l', 42: 'm', 43: 'n', 44: 'o', 45: 'p', 46: 'q', 47: 'r', 48: 's', 49: 't', 50: 'u', 51: 'v', 52: 'w', 53: 'x', 54: 'y', 55: 'z', 56: 'æ', 57: '—', 58: '‘', 59: '’', 60: '“', 61: '”', 62: '•', 63: '™'}



#interate from 0 to total lenghth of input text - window length and mention steps as in 1 step at a time
for i in range(0, len(raw_text)-seq_length, 1):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([char_to_ind[char] for char in seq_in])
    dataY.append(char_to_ind[seq_out])


#total how many patterns as character lists(now converted into interger lists) we have for training
n_patterns = len(dataX)

#reshape the normal list we have to suit the model
X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(len(chars))
y = to_categorical(dataY, num_classes=len(char_to_ind))

#create sequential model
# model = Sequential()

model = load_model("AI/Model/shakespeare_generator.keras")

# #Build the model
# #LSTM with 256 memory cells to learn from the input data 
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))

# model.add(LSTM(256))
# model.add(Dropout(0.2))

# #final decision maker layer which will give score for each possible character and that score will be converted to probability
# model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#training the model on 5 epoched and batch size of 128 meaning update after processing 128 sequences
model.fit(X, y, epochs=3 , batch_size=128)

#save the model for future use
model.save("AI/Model/shakespeare_generator.keras")

#select the Seed randomely from the training data to test the model
# start = np.random.randint(0, len(dataX)-1)
# pattern = dataX[start]
pattern = [char_to_ind[char] for char in "i love you so much to the point where i can t even describe in the words and i can do anythinng love"]

#loop 500 times meaning genererate 500 words that will follow seed
for i in range(500):
    #reshape the input pattern (list of 100 characters)
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))

    #output the probability of all characters and save the item with most proab in index to print result on console
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    sys.stdout.write(result)

    #finally update the pattern
    pattern.append(index)
    pattern = pattern[1:len(pattern)]