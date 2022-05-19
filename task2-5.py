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
            
            if row_id == 0:
                header.append(start_sec)

            try:
                end_sec = int(row[1])
            except:
                end_sec = 10 * int(row[1].split('.')[0]) + int(row[1].split('.')[1][0])   

            if row_id == len(csv_list) - 1:
                header.append(end_sec)
            ans.append([start_sec, end_sec])
        ans.insert(0, header)
        ans_list.append(ans)

    # print(ans_list) 

for file_id, filename in enumerate(files):
        
    ans_song_list = ans_list[file_id][1:]
    ans_cur = 1
    cur_start, cur_end = ans_song_list[0][0], ans_song_list[0][1]
    # print('start, end:', cur_start, ',',cur_end)
    estimate_list = []

    midi_data = pretty_midi.PrettyMIDI('{folder}/{file}'.format(folder = folder_path, file = filename))
    f = midi_data.get_chroma()

    num_bin, num_frame = f.shape[0], min(f.shape[1], cur_end - cur_start + 1)

    last_tonic_str = '' 
    seg_start, seg_end = cur_start, cur_start

    for frame in range(cur_start, min(cur_end + 1, cur_start + num_frame)):
        
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
        for i,ks in enumerate(template.KS_major_template):
            r = scipy.stats.pearsonr(bin_avg, ks)[0]
            if r > max_r or i == 0:
                max_r = r
                tonic_str = template.tonic[i] + ' ' + template.tonic[-2]

        for i,ks in enumerate(template.KS_minor_template):
            r = scipy.stats.pearsonr(bin_avg, ks)[0]
            if r > max_r:
                max_r = r
                tonic_str = template.tonic[i] + ' ' + template.tonic[-1]
        
        # print('last:', last_tonic_str)
        #print('curr:', tonic_str)

        if frame == cur_start:
            last_tonic_str = tonic_str
        
        if tonic_str != last_tonic_str:
            seg_end = frame
            estimate_list.append([seg_start/10, seg_end/10])
            seg_start = seg_end + 1
        
        last_tonic_str = tonic_str
    
    if seg_start < min(cur_end + 1, cur_start + num_frame) - 1:
        estimate_list.append([seg_start/10, (min(cur_end + 1, cur_start + num_frame) - 1)/10])
    
    score_u = mir_eval.chord.underseg(np.array(ans_song_list), np.array(estimate_list))
    score_o = mir_eval.chord.overseg(np.array(ans_song_list), np.array(estimate_list))
    score_a = mir_eval.chord.seg(np.array(ans_song_list), np.array(estimate_list))
    
    print('| file{:02d} | {} |'.format(file_id, filename))
    print('| ---- | ---- |')
    print('| underseg | {} |'.format(score_u))
    print('| overseg | {} |'.format(score_o))
    print('| meanseg | {} |'.format(score_a))
    print()
    
    