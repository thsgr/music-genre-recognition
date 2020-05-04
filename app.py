from flask import Flask, render_template, redirect, url_for, session, request, flash
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import io
import sys
import os
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## IA

def loading_the_model(model_path):
    global model
    model = load_model(model_path)


model_path = 'first_model.model'
loading_the_model(model_path)


def prepare_audio(audio):
    # Transforme une piste audio en une liste de 26 valeurs
    y, sr = librosa.load(audio, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    L = []
    L.append(np.mean(chroma_stft))
    L.append(np.mean(rmse))
    L.append(np.mean(spec_cent))
    L.append(np.mean(spec_bw))
    L.append(np.mean(rolloff))
    L.append(np.mean(zcr))
    for e in mfcc:
        L.append(f' {np.mean(e)}')
    return np.array(L, dtype = float)

dic_genre = {0:"blues", 1:"classical", 2:"country", 3:"disco", 4:"hiphop", 5:"jazz", 6:"metal", 7:"pop", 8:"reggae", 9:"rock"}

def normalization(x):
    # Normalise un vecteur comme ceux normalisés pour l'IA
    scaler = StandardScaler()
    data = pd.read_csv('data.csv')
    data = data.drop(['filename'],axis=1)
    a = np.array(data.iloc[:, :-1], dtype = float)
    x = np.expand_dims(x, axis=0)
    x = np.concatenate((x, a))
    x = scaler.fit_transform(x)
    return x[0]

def process_predictions(L):
    # Prédit le genre selon notre modèle
    L = normalization(L)
    L = np.expand_dims(L, axis=0)
    Y_pred = model.predict(L)
    genre = dic_genre[np.argmax(Y_pred[0])]
    return genre


# Index
@app.route('/')
def index():
    return render_template('index.html')


#Style CSS

#@app.route('/')
#def style():
#    return

#about
@app.route('/about')
def about():
    return render_template('about.html')


#project
@app.route('/project')
def project():
    return render_template('project.html')


@app.route('/upload')
def upload_file1():
   return render_template('upload.html')


# Coupe une musique et la met au bon format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=20000)
    if filename[-3:] == 'wav':
        segment = AudioSegment.from_wav(filename)
    else:
        segment = AudioSegment.from_mp3(filename)
    longueur = len(segment)
    if longueur<19999:
        # musique de moins de 20 secondes ne peut être analysé
        return render_template('uploader/mauvaisfichier.html')
    else :
        n = int((longueur-20000)/2) #20 secondes au milieu de la musique
        segment = segment[n:n+20000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename[:-3]+'wav', format='wav')
    return longueur

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename[-3:] == 'mp3' or f.filename[-3:] == 'wav':
            f.save(secure_filename(f.filename))
            nom = f.filename[:-4]
            name = f.filename.replace(' ','_')
            longueur = int(preprocess_audio(name)//1000) #longueur de la musique (en ms)
            minutes = longueur//60
            sec = longueur%60
            audio = name[:-3]+'wav'
            L = prepare_audio(audio)
            genre = process_predictions(L)
            if f.filename[-3:] == 'mp3':
                os.remove(audio)
            os.remove(name)
            return render_template(genre +'.html', genre=genre, nom=nom, minutes=minutes, sec=sec)
        else:
            # ce n'est pas une musique mp3 ou wav, on ne télécharge pas le fichier
            return render_template('uploader/mauvaisfichier.html')
        

if __name__ == '__main__':
    app.run(debug=True)


