import mir_eval
import pretty_midi
import scipy
import numpy as np
from os import listdir
import template
import csv

print('- [{}]'.format('midi'))
    
folder_path = "SWD/01_RawData/score_midi"
files = listdir(folder_path)

score = 0
rawscore = 0
num_of_file = len(files)

ans_list = []
with open('SWD/02_Annotations/ann_score_globalkey_2.csv') as anscsv:
    csvreader = csv.reader(anscsv)
    for row in csvreader:
        ans_list.append(row[1])

for file_id, f in enumerate(files):
   
    midi_data = pretty_midi.PrettyMIDI('{folder}/{file}'.format(folder = folder_path, file = f))
    # print(midi_data.instruments) 

    feature_chroma = midi_data.get_chroma()
    # print('chroma: ({}, {})'.format(feature_chroma.shape[0], feature_chroma.shape[1]))
    # print(feature_chroma)
        
    bin_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_bin, num_frame = feature_chroma.shape[0], feature_chroma.shape[1]
    max_bin, max_r = 0, 0

    for i in range(num_bin):
        for energy in feature_chroma[i]:
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

    score += mir_eval.key.evaluate(ans_list[file_id], tonic_str)['Weighted Score']
    if ans_list[file_id] == tonic_str:
        rawscore += 1
    
print('score: {}'.format(score/num_of_file))
print('raw score: {}'.format(rawscore/num_of_file))
    