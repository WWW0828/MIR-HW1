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
raw_scores_stft = []
raw_scores_cqt = []
raw_scores_cens = []

for v_id, v in enumerate(versions):

    print('- [{}]'.format(v))
    
    folder_path = "SWD/01_RawData/audio_wav/" + v
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    raw_score_stft, raw_score_cqt, raw_score_cens = 0, 0, 0
    num_of_file = len(files)

    ans_list = []
    with open('SWD/02_Annotations/ann_audio_globalkey_' + v + '.csv') as anscsv:
        csvreader = csv.reader(anscsv)
        for row in csvreader:
            ans_list.append(row[2])

    for file_id, f in enumerate(files):
   
        y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = f))
        # print('file{:02d}: {}'.format(file_id, f))
        
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
                score_stft += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
                if ans_list[file_id] == tonic_str:
                    raw_score_stft += 1
            elif id == 1:
                score_cqt += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
                if ans_list[file_id] == tonic_str:
                    raw_score_cqt += 1
            else:
                score_cens += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
                if ans_list[file_id] == tonic_str:
                    raw_score_cens += 1
            
    scores_stft.append(score_stft/num_of_file)
    scores_cqt.append(score_cqt/num_of_file)
    scores_cens.append(score_cens/num_of_file)  
    raw_scores_stft.append(raw_score_stft/num_of_file)
    raw_scores_cqt.append(raw_score_cqt/num_of_file)
    raw_scores_cens.append(raw_score_cens/num_of_file)   
    
    print('| Raw Score      | {:.6f} | {:.6f} | {:.6f} |'.format(raw_scores_stft[v_id], raw_scores_cqt[v_id], raw_scores_cens[v_id]))
    print('| Weighted Score | {:.6f} | {:.6f} | {:.6f} |'.format(scores_stft[v_id], scores_cqt[v_id], scores_cens[v_id]))
    