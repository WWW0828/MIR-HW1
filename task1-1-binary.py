import librosa
import mir_eval
import scipy
import numpy as np
from os import listdir
import template

genres = ["blues", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

scores_stft = []
scores_cqt = []
scores_cens = []

for g in genres:

    folder_path = "GTZAN/wav/" + g
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    num_of_file = len(files)

    for file_id, f in enumerate(files):

        print('file: {:03d}'.format(file_id))
        skip = False
        ans_str = ''
        with open('GTZAN/key/{genre}/{genre}.{id:05d}.lerch.txt'.format(genre = g, id = file_id)) as anstxt:
            ans_tonic = int(anstxt.readline())
            if ans_tonic not in range(24):
                skip = True
                print('skip')
                num_of_file -= 1
            else:
                ans_str = template.eval_tonic[ans_tonic]
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
            for i,bt in enumerate(template.binary_template):
                r = scipy.stats.pearsonr(bin_avg, bt)[0]
                if r > max_r or i == 0:
                    max_r = r
                    max_bin = i

            tonic_str = ''
            # check energy of M and m
            if bin_avg[template.minor[max_bin]] > bin_avg[max_bin]:
                tonic_str = template.tonic[template.minor[max_bin]] + ' ' + template.tonic[-1]
            else:
                tonic_str = template.tonic[max_bin] + ' ' + template.tonic[-2]

            if id == 0:
                print('stft:', tonic_str, end = ', ')
                score_stft += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
            elif id == 1:
                print(' cqt:', tonic_str, end = ', ')
                score_cqt += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
            else:
                print('cens:', tonic_str, end = '')
                score_cens += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
            
        print()
        print('score: stft {}, cqt {}, cens {}'.format(score_stft, score_cqt, score_cens))

    scores_stft.append(score_stft/num_of_file)
    scores_cqt.append(score_cqt/num_of_file)
    scores_cens.append(score_cens/num_of_file)   
    print(g, ':')
    print(scores_stft, scores_cqt, scores_cens)            
    break