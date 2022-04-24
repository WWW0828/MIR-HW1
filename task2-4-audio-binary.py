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

time_window = 300   # 0.1 sec

for v_id, v in enumerate(versions):

    print('- [{}]'.format(v))
    
    folder_path = "SWD/01_RawData/audio_wav/" + v
    files = listdir(folder_path)

    score_stft, score_cqt, score_cens = 0, 0, 0
    num_of_frame = 0

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
                else:
                    # print(row_id)
                    pass

                ans.append([start_sec, row[2]])

            ans.insert(0, header)
            ans_list.append(ans)

    # print(ans_list) 

    for file_id, filename in enumerate(files):
        
        ans_song_list = ans_list[file_id]
        ans_cur = 1
        cur_start, cur_end = ans_song_list[0][0], ans_song_list[0][1]

        print('start: {}, end: {}'.format(cur_start, cur_end))

        y, sr = librosa.load('{folder}/{file}'.format(folder = folder_path, file = filename))
        
        feature_stft = librosa.feature.chroma_stft(y = y, sr = sr, hop_length = 2205)
        feature_cqt = librosa.feature.chroma_cqt(y = y, sr = sr, hop_length = 2205)
        feature_cens = librosa.feature.chroma_cens(y = y, sr = sr, hop_length = 2205)

        features = [feature_stft, feature_cqt, feature_cens]
        

        for id,f in enumerate(features):
            
            num_bin, num_frame = f.shape[0], min(f.shape[1], cur_end - cur_start + 1)
            
            for frame in range(cur_start, min(cur_end + 1, cur_start + num_frame)):

                num_of_frame += 1
                
                bin_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                if frame >= 150:
                    frame_start = frame - 150
                else:
                    frame_start = 0

                if frame < f.shape[1] - 150:
                    frame_end = frame + 150
                else:
                    frame_end = f.shape[1] - 1
                
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

                if ans_cur < len(ans_song_list) - 1:
                    if ans_song_list[ans_cur + 1][0] <= frame:
                        ans_cur += 1
                answer_str = ans_song_list[ans_cur][1]

                if id == 0:
                    score_stft += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
                elif id == 1:
                    score_cqt += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
                else:
                    score_cens += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
    
    scores_stft.append(score_stft/num_of_frame)
    scores_cqt.append(score_cqt/num_of_frame)
    scores_cens.append(score_cens/num_of_frame)   
    
    print('score: stft {}, cqt {}, cens {}'.format(scores_stft[v_id], scores_cqt[v_id], scores_cens[v_id]))
    