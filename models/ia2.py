import numpy as np
import random
from pydub import AudioSegment
"""
import sys
import io
import os
"""
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import tensorflow.keras as keras

"""
cd C:\\Users\\acarr\\Documents\\Cours\\CentraleSupelec\\CodingWeeks\\codingweeks2\\reconnaissance_genre
"""

"""
# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
"""

# We use plt.specgram to get the spectrograms
# Notice if the audio data have two channels we only use one given that they are equal
def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

"""
audio_file = "./genres/blues/blues.00000.wav"
_, data = wavfile.read(audio_file)
x = graph_spectrogram(audio_file)
print("Time steps in audio recording before spectrogram", data.shape[0])
print("Time steps in input after spectrogram", x.shape)


Time steps in audio recording before spectrogram 661794
Time steps in input after spectrogram (101, 8270)
"""

# Load raw audio files for speech synthesis
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
dic_genres = {}
for k in range(len(genres)):
    dic_genres[genres[k]] = k

## Preprocess the audio to the correct format
for g in genres:
    print(f'{g}')
    for audio_file in glob.glob(f'./genres/{g}/*.wav'):
        audio_file = '.\\genres\\'+audio_file[9:]
        # Trim or pad audio segment to 20000ms
        padding = AudioSegment.silent(duration=20000)
        segment = AudioSegment.from_wav(audio_file)[:20000]
        segment = padding.overlay(segment)
        # Set frame rate to 44100
        segment = segment.set_frame_rate(44100)
        # Export as wav
        segment.export(audio_file, format='wav')

##

Tx = 11023 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

##

for g in genres:
    X = []
    Y = []
    print(f'{g}')
    for audio_file in glob.glob(f'./genres/{g}/*.wav'):
        lengenre = len(f'{g}')
        filename = audio_file[10+lengenre:]
        audio_file = '.\\genres\\'+audio_file[9:]
        _, data = wavfile.read(audio_file)
        x = graph_spectrogram(audio_file)
        x = x.transpose()
        X.append(x)
        Y.append(dic_genres[f'{g}'])
    X = np.array(X)
    Y = np.array(Y)
    np.save(f"./dataset/X_{g}.npy", X)
    np.save(f"./dataset/Y_{g}.npy", Y)


##

X = np.load(f"./dataset/X_blues.npy")
Y = np.load(f"./dataset/Y_blues.npy")
for g in genres[1:]:
    print(f'{g}')
    a = np.load(f"./dataset/X_{g}.npy")
    b = np.load(f"./dataset/Y_{g}.npy")
    X = np.concatenate((X,a))
    Y = np.concatenate((Y,b))
np.save("./dataset/X.npy", X)
np.save("./dataset/Y.npy", Y)

##

X = np.load("./dataset/X.npy")
Y = np.load("./dataset/Y.npy")

r = random.random()
random.shuffle(X, lambda: r)
random.shuffle(Y, lambda: r)

X_train, Y_train = X[:900], Y[:900]
X_test, Y_test = X[900:], Y[900:]


"""
Normalise ?
"""
X_train = keras.utils.normalize(X_train)
X_test = keras.utils.normalize(X_test)

np.save("./dataset/X_train.npy", X_train)
np.save("./dataset/Y_train.npy", Y_train)
np.save("./dataset/X_test.npy", X_test)
np.save("./dataset/Y_test.npy", Y_test)

##

#from tensorflow.keras.optimizers import Adam

model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


#opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

##

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=196, kernel_size=15, strides=4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

"""
def model(input_shape):


    X_input = Input(shape = input_shape)

    # CONV layer
    X = X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = Dense(128, activation='relu')(X)
    X = Dense(128, activation='relu')(X)
    X = (X)

    model = Model(inputs = X_input, outputs = X)

    return model


model = model(input_shape = (Tx, n_freq))
"""

# Create an optimizer (use Adam), and pass the parameters (learning rate, momentum values and decay rate)
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
# Compile the model, use loss binary CE and the metric as accuracy
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

##

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=196, kernel_size=15, strides=4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

##

X_train = np.load(f"./dataset/X_train.npy")
Y_train = np.load(f"./dataset/Y_train.npy")
X_test = np.load(f"./dataset/X_test.npy")
Y_test = np.load(f"./dataset/Y_test.npy")

model.fit(X_train, Y_train, epochs=3)

model.summary()

##

val_loss, val_acc = model.evaluate(X_test, Y_test)
print(f'Validation loss: {val_acc}, Validation accuracy: {val_acc}')

#model.save('first_model_of_many_to_come.model')
#new_model = keras.models.load_model('first_model_of_many_to_come.model')


##

Y_pred = model.predict(X_test[1:2])

print(np.argmax(Y_pred[0]))
print(Y_test[1])
