import sys
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#open the plain text file in read mode and save contents in raw_text
with open("AI\shakespear.txt", "r", encoding='utf-8') as file:
    raw_text = file.read()

#keep entire text document in lowercase for ease
raw_text = raw_text.lower()


#defining the window size and empty lists which will be used to prepare model later
seq_length = 100
dataX = []
dataY = []

#Create uniq list of all charcters used in text and than create dictionaries for accessing char by index and index by characters
chars = sorted(list(set(raw_text)))
char_to_ind = dict((char,index) for index, char in enumerate(chars))
int_to_char = dict((index,char) for index,char in enumerate(chars))

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
y = to_categorical(dataY)

#create sequential model
model = Sequential()

#Build the model
#LSTM with 256 memory cells to learn from the input data 
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256))
model.add(Dropout(0.2))

#final decision maker layer which will give score for each possible character and that score will be converted to probability
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#training the model on 5 epoched and batch size of 128 meaning update after processing 128 sequences
model.fit(X, y, epochs=5, batch_size=128)

#save the model for future use
model.save("shakespeare_generator.h5")

#select the Seed randomely from the training data to test the model
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

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