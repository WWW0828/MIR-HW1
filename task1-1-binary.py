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
raw_scores_stft = []
raw_scores_cqt = []
raw_scores_cens = []

for g_id, g in enumerate(genres):

    print('- [{}]'.format(g))
    
    folder_path = "GTZAN/wav/" + g
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    raw_score_stft, raw_score_cqt, raw_score_cens = 0, 0, 0
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

        y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = f))
        
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
                score_stft += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
                if ans_str == tonic_str:
                    raw_score_stft += 1
            elif id == 1:
                score_cqt += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
                if ans_str == tonic_str:
                    raw_score_cqt += 1
            else:
                score_cens += mir_eval.key.evaluate(ans_str, tonic_str)['Weighted Score']
                if ans_str == tonic_str:
                    raw_score_cens += 1
    
    scores_stft.append(score_stft/num_of_file)
    scores_cqt.append(score_cqt/num_of_file)
    scores_cens.append(score_cens/num_of_file)   
    raw_scores_stft.append(raw_score_stft/num_of_file)
    raw_scores_cqt.append(raw_score_cqt/num_of_file)
    raw_scores_cens.append(raw_score_cens/num_of_file)   
        
    print('wei score: | {} | {:.6f} | {:.6f} | {:.6f} |'.format(g, scores_stft[g_id], scores_cqt[g_id], scores_cens[g_id]))
    print('raw score: | {} | {:.6f} | {:.6f} | {:.6f} |'.format(g, raw_scores_stft[g_id], raw_scores_cqt[g_id], raw_scores_cens[g_id]))
