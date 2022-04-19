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

for g_id, g in enumerate(genres):

    print('- [{}]'.format(g))

    folder_path = "GTZAN/wav/" + g
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    num_of_file = len(files)

    for file_id, f in enumerate(files):

        skip = False
        ans_str = ''
        with open('GTZAN/key/{genre}/{genre}.{id:05d}.lerch.txt'.format(genre = g, id = file_id)) as anstxt:
            ans_tonic = int(anstxt.readline())
            if ans_tonic not in range(24):
                skip = True
                num_of_file -= 1
            else:
                ans_str = template.eval_tonic[ans_tonic]
        
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
            for i,ks in enumerate(template.HM_major_template):
                r = scipy.stats.pearsonr(bin_avg, ks)[0]
                if r > max_r or i == 0:
                    max_bin = i
                    max_r = r
                    tonic_str = template.tonic[i] + ' ' + template.tonic[-2]

            for i,ks in enumerate(template.HM_minor_template):
                r = scipy.stats.pearsonr(bin_avg, ks)[0]
                if r > max_r or (r == max_r and bin_avg[i] > bin_avg[max_bin]):
                    max_bin = i
                    max_r = r
                    tonic_str = template.tonic[i] + ' ' + template.tonic[-1]


            if id == 0:
                score_stft += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
            elif id == 1:
                score_cqt += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
            else:
                score_cens += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
    
    scores_stft.append(score_stft/num_of_file)
    scores_cqt.append(score_cqt/num_of_file)
    scores_cens.append(score_cens/num_of_file)   
        
    print('score: stft {}, cqt {}, cens {}'.format(scores_stft[g_id], scores_cqt[g_id], scores_cens[g_id]))
