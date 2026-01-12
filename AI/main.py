import sys
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


with open("AI\shakespear.txt", "r", encoding='utf-8') as file:
    raw_text = file.read()

raw_text = raw_text.lower()

seq_length = 100
dataX = []
dataY = []

chars = sorted(list(set(raw_text)))
char_to_ind = dict((char,index) for index, char in enumerate(chars))
int_to_char = dict((index,char) for index,char in enumerate(chars))

for i in range(0, len(raw_text)-seq_length, 1):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)

n_patterns = len(dataX)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(len(chars))

y = to_categorical(dataY)

model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256))
model.add(Dropout(0.2))

model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=5, batch_size=128)

model.save("shakespeare_generator.h5")

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

for i in range(500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    sys.stdout.write(result)
    
    pattern.append(index)
    pattern = pattern[1:len(pattern)]