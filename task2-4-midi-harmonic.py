import pretty_midi
import mir_eval
import scipy
import numpy as np
from os import listdir
import template
import csv

scores = []
raw_scores = []

time_window = 150   # 0.1 sec

folder_path = "SWD/01_RawData/score_midi"
files = listdir(folder_path)

score = 0
raw_score = 0

num_of_frame = 0

ann1_folder = 'SWD/02_Annotations/ann_score_localkey-time-ann1'
ann1_files = listdir(ann1_folder)
ans_list = []

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

    midi_data = pretty_midi.PrettyMIDI('{folder}/{file}'.format(folder = folder_path, file = filename))
    f = midi_data.get_chroma()

    num_bin, num_frame = f.shape[0], min(f.shape[1], cur_end - cur_start + 1)
            
    for frame in range(cur_start, min(cur_end + 1, cur_start + num_frame)):
                
            # print('frame:', frame)
                
        num_of_frame += 1
                
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
        
        tonic_str = ''
        for i,ks in enumerate(template.HM_major_template):
            r = scipy.stats.pearsonr(bin_avg, ks)[0]
            if r > max_r or i == 0:
                max_r = r
                tonic_str = template.tonic[i] + ' ' + template.tonic[-2]

        for i,ks in enumerate(template.HM_minor_template):
            r = scipy.stats.pearsonr(bin_avg, ks)[0]
            if r > max_r or (r == max_r and bin_avg[i] > bin_avg[max_bin]):
                max_r = r
                tonic_str = template.tonic[i] + ' ' + template.tonic[-1]
                    
        if ans_cur < len(ans_song_list) - 1:
            if ans_song_list[ans_cur + 1][0] <= frame:
                ans_cur += 1

        answer_str = ans_song_list[ans_cur][1]
                
        score += mir_eval.key.evaluate(answer_str, tonic_str)['Weighted Score']
        if answer_str == tonic_str:
            raw_score += 1
                
   
print('wei score: {:.6f}'.format(score/num_of_frame))
print('raw score: {:.6f}'.format(raw_score/num_of_frame))
