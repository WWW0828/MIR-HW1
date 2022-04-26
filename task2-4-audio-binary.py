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

time_window = 150   # 0.1 sec

for v_id, v in enumerate(versions):

    print('- [{}]'.format(v))
    
    folder_path = "SWD/01_RawData/audio_wav/" + v
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    raw_score_stft, raw_score_cqt, raw_score_cens = 0, 0, 0
    
    num_of_frame = [0, 0, 0]

    ann1_folder = 'SWD/02_Annotations/ann_audio_localkey-ann1-' + v
    ann1_files = listdir(ann1_folder)
    ans_list = []

    # [ans_list]: [[song], [song], [song], ...]
    # [song]: [[start, end], [frame], [frame], ...]s
    # [frame]: [start_sec, 'major/minor']

    for f in ann1_files:
        
        ans = []

        with open('{folder}/{file}'.format(folder = ann1_folder, file = f)) as anscsv:

            csvreader = csv.reader(anscsv)
            csv_list = list(csvreader)
            
            header = []
            
            for row_id, row in enumerate(csv_list):
                
                try:
                    start_sec = int(row[0])
                except:
                    start_sec = 10 * int(row[0].split('.')[0]) + int(row[0].split('.')[1][0])
                    if len(row[0].split('.')[1]) > 1:
                        start_sec += 1
                
                if len(header) == 0:
                    header.append(start_sec)

                if row_id == len(csv_list) - 1:
                    try:
                        end_sec = int(row[1])
                    except:
                        end_sec = 10 * int(row[1].split('.')[0]) + int(row[1].split('.')[1][0])   
                    header.append(end_sec)

                ans.append([start_sec, row[2]])

            ans.insert(0, header)
            ans_list.append(ans)

    # print(ans_list) 

    for file_id, filename in enumerate(files):
        
        ans_song_list = ans_list[file_id]
        ans_cur = 1
        cur_start, cur_end = ans_song_list[0][0], ans_song_list[0][1]

        # print('start: {}, end: {}'.format(cur_start, cur_end))
        # print(ans_song_list)

        y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = filename))
        
        feature_stft = librosa.feature.chroma_stft(y = y, sr = sr, hop_length = 2205)
        feature_cqt = librosa.feature.chroma_cqt(y = y, sr = sr, hop_length = 2205)
        feature_cens = librosa.feature.chroma_cens(y = y, sr = sr, hop_length = 2205)

        features = [feature_stft, feature_cqt, feature_cens]
        

        for id,f in enumerate(features):
            
            num_bin, num_frame = f.shape[0], min(f.shape[1], cur_end - cur_start + 1)
            
            for frame in range(cur_start, min(cur_end + 1, cur_start + num_frame)):
                
                # print('frame:', frame)
                
                num_of_frame[id] += 1
                
                bin_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                if frame >= time_window:
                    frame_start = frame - time_window
                else:
                    frame_start = 0

                if frame < f.shape[1] - time_window:
                    frame_end = frame + time_window
                else:
                    frame_end = f.shape[1] - 1
                
                # print('start:', frame_start, ', end:', frame_end)

                for i in range(frame_start, frame_end + 1):
                    for j in range(num_bin):
                        bin_avg[j] += f[j][i]

                max_bin, max_r = 0, 0

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

                flag = False
                if ans_cur < len(ans_song_list) - 1:
                    if ans_song_list[ans_cur + 1][0] <= frame:
                        ans_cur += 1
                        flag = True

                answer_str = ans_song_list[ans_cur][1]

                if id == 0:
                    score_stft += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
                    if answer_str == tonic_str:
                        raw_score_stft += 1
                elif id == 1:
                    score_cqt += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
                    if answer_str == tonic_str:
                        raw_score_cqt += 1
                else:
                    score_cens += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
                    if answer_str == tonic_str:
                        raw_score_cens += 1

    scores_stft.append(score_stft/num_of_frame[0])
    scores_cqt.append(score_cqt/num_of_frame[1])
    scores_cens.append(score_cens/num_of_frame[2]) 
    raw_scores_stft.append(raw_score_stft/num_of_frame[0])
    raw_scores_cqt.append(raw_score_cqt/num_of_frame[1])
    raw_scores_cens.append(raw_score_cens/num_of_frame[2])   
    
    print('wei score: | {} | {:.6f} | {:.6f} | {:.6f} |'.format(v, scores_stft[v_id], scores_cqt[v_id], scores_cens[v_id]))
    print('raw score: | {} | {:.6f} | {:.6f} | {:.6f} |'.format(v, raw_scores_stft[v_id], raw_scores_cqt[v_id], raw_scores_cens[v_id]))
    