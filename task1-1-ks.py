import librosa
import mir_eval
import scipy
import numpy as np
from os import listdir
import template

genres = ["blues", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

minor = {0: 9, 1: 10, 2: 11, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
major = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 0, 10: 1, 11: 2}

tonic = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'major', 'minor']
eval_tonic = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'major', 'minor']



scores_stft = []
scores_cqt = []
scores_cens = []

for g in genres:

    folder_path = "GTZAN/wav/" + g
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    num_of_file = len(files)

    for file_id, f in enumerate(files):

        print('file:{:03d}'.format(file_id))
        skip = False
        ans_str = ''
        with open('GTZAN/key/{genre}/{genre}.{id:05d}.lerch.txt'.format(genre = g, id = file_id)) as anstxt:
            ans_tonic = int(anstxt.readline())
            if ans_tonic not in range(24):
                skip = True
                print('skip')
                num_of_file -= 1

            if ans_tonic >= 12:
                ans_str = eval_tonic[ans_tonic - 12] + ' ' + eval_tonic[-1]
            else:
                ans_str = eval_tonic[ans_tonic] + ' ' + eval_tonic[-2]
            print('ans:', ans_str)
        
        if skip:
            continue

        y, sr = librosa.load(folder_path + '/' + f)
        
        feature_stft = librosa.feature.chroma_stft(y = y, sr = sr)
        feature_cqt = librosa.feature.chroma_cqt(y = y, sr = sr)
        feature_cens = librosa.feature.chroma_cens(y = y, sr = sr)
        features = [feature_stft, feature_cqt, feature_cens]

        for id,f in enumerate(features):
            
            bin_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            num_bin, num_frame = f.shape[0], f.shape[1]
            max_bin, max_r = 0, 0

            for i in range(num_bin):
                for energy in f[i]:
                    bin_avg[i] += energy
                bin_avg[i] /= num_frame
            
            # calculate R
            # find biggest R

            tonic_str = ''
            for i,bt in enumerate(template.KS_major_template):
                r = scipy.stats.pearsonr(bin_avg, bt)[0]
                if r > max_r or i == 0:
                    tonic_str = tonic[i] + ' ' + tonic[-2]

            for i,bt in enumerate(template.KS_minor_template):
                r = scipy.stats.pearsonr(bin_avg, bt)[0]
                if r > max_r:
                    tonic_str = tonic[i] + ' ' + tonic[-1]


            if id == 0:
                print('stft:', tonic_str, end = ', ')
                score_stft += mir_eval.key.evaluate(tonic_str, ans_str)['Weighted Score']
            elif id == 1:
                print(' cqt:', tonic_str, end = ', ')
                score_cqt += mir_eval.key.evaluate(tonic_str, ans_str)['Weighted Score']
            else:
                print('cens:', tonic_str, end = '')
                score_cens += mir_eval.key.evaluate(tonic_str, ans_str)['Weighted Score']
            
        print()
        print('score: stft:{}, cqt:{}, cens:{}'.format(score_stft, score_cqt, score_cens))

    scores_stft.append(score_stft/num_of_file)
    scores_cqt.append(score_cqt/num_of_file)
    scores_cens.append(score_cens/num_of_file)   
    print(g, ':')
    print(scores_stft, scores_cqt, scores_cens)            
    break