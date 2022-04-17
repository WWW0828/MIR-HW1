import librosa
import scipy
import numpy as np
from os import listdir
genres = ["blues", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# librosa.feature.chroma_stft
# librosa.feature.chroma_cqt
# librosa.feature.chroma_cens
# return normalized energy for each chroma bin at each frame.

for g in genres:
    folder_path = "GTZAN/wav/" + g
    files = listdir(folder_path)
    for f in files:
        y, sr = librosa.load(folder_path + '/' + f)
        
        feature_stft = librosa.feature.chroma_stft(y = y, sr = sr)
        feature_cqt = librosa.feature.chroma_cqt(y = y, sr = sr)
        feature_cens = librosa.feature.chroma_cens(y = y, sr = sr)
        
        print('stft:', feature_stft.shape)
        print(feature_stft)
        print('cqt:', feature_cqt.shape)
        print(feature_cqt)
        print('cens:', feature_cens.shape)
        print(feature_cens)

        break
    break