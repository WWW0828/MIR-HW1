import librosa
import scipy
import numpy as np
from os import listdir

genres = ["blues", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
minor = {0: 9, 1: 10, 2: 11, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
major = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 0, 10: 1, 11: 2}

#   0   1  2   3  4  5   6  7   8  9  10  11
# [ C, C#, D, D#, E, F, F#, G, G#, A, A#, B ]
binary_template = [[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                   [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                   [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]]
for g in genres:
    folder_path = "GTZAN/wav/" + g
    files = listdir(folder_path)
    result_stft = []
    result_cqt = []
    result_cens = []
    
    for f in files:
        y, sr = librosa.load(folder_path + '/' + f)
        
        feature_stft = librosa.feature.chroma_stft(y = y, sr = sr)
        feature_cqt = librosa.feature.chroma_cqt(y = y, sr = sr)
        feature_cens = librosa.feature.chroma_cens(y = y, sr = sr)
        
        features = [feature_stft, feature_cqt, feature_cens]

        for id,f in enumerate(features):
            
            bin_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            num_bin = f.shape[0]
            num_frame = f.shape[1]
            
            max_bin = 0
            max_r = -1

            for i in range(num_bin):
                for energy in f[i]:
                    bin_avg[i] += energy
                bin_avg[i] = bin_avg[i] / num_frame
            

            # calculate R
            # find biggest R
            for i,bt in enumerate(binary_template):
                r = scipy.stats.pearsonr(bin_avg, bt)[0]
                if r > max_r or i == 0:
                    max_r = r
                    max_bin = i


            # check energy of M and m
            if bin_avg[minor[max_bin]] > bin_avg[max_bin]:
                max_bin = minor[max_bin]
                max_bin = (max_bin + 3) % 12
                max_bin += 12
            else:
                max_bin = (max_bin + 3) % 12

            if id == 0:
                result_stft.append(max_bin)
                # print('stft:', max_bin)
            elif id == 1:
                result_cqt.append(max_bin)
                # print('cqt:', max_bin)
            else:
                result_cens.append(max_bin)
                # print('cens:', max_bin)

    print('stft:', result_stft[0:10])
    print('cqt:', result_cqt[:10])
    print('cens:', result_cens[:10])
        
    break