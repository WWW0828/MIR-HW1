import librosa
import mir_eval
import scipy
import numpy as np
from os import listdir
import template
import csv

versions = ['FI66', 'FI80', 'HU33', 'SC06']

scores_stft = []
scores_cqt = []
scores_cens = []

for v_id, v in enumerate(versions):

    print('- [{}]'.format(v))
    
    folder_path = "SWD/01_RawData/audio_wav/" + v
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    num_of_file = len(files)

    ans_list = []
    with open('SWD/02_Annotations/ann_audio_globalkey_' + v + '.csv') as anscsv:
        csvreader = csv.reader(anscsv)
        for row in csvreader:
            ans_list.append(row[2])

    for file_id, f in enumerate(files):
   
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
                score_stft += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
            elif id == 1:
                score_cqt += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
            else:
                score_cens += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
            
    scores_stft.append(score_stft/num_of_file)
    scores_cqt.append(score_cqt/num_of_file)
    scores_cens.append(score_cens/num_of_file)   
    
    print('score: stft {}, cqt {}, cens {}'.format(scores_stft[v_id], scores_cqt[v_id], scores_cens[v_id]))
    