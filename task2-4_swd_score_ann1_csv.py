import csv
import template
from os import listdir

localkey_folder = "SWD/02_Annotations/ann_score_localkey-ann1"
localkey_files = listdir(localkey_folder)

measure2time_folder = "time_per_measure_ann_score_localkey-ann1"
destination_folder = "SWD/02_Annotations/ann_score_localkey-time-ann1"
for file in localkey_files:
    
    measure2time_list = []
    with open('{}/{}'.format(measure2time_folder, file)) as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        for row in csvreader:
            measure2time_list.append(row[1])
    
    rows = []
    with open('{}/{}'.format(localkey_folder, file)) as f:
        # 1.000;15.750;"D:min"
        csvreader = csv.reader(f)
        header = next(csvreader)
        for row in csvreader:
            newrow = row[0].split(';')
            newrow[2] = newrow[2].split('"')[1].split(':')
            if newrow[2][0][-1] == 'b':
                newrow[2] = template.flat2sharp[newrow[2][0]] + ' ' + template.flat2sharp[newrow[2][1]]
            else:
                newrow[2] = newrow[2][0] + ' ' + template.flat2sharp[newrow[2][1]]
            rows.append(newrow)


    with open('{}/{}'.format(destination_folder, file), 'w', newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(rows)

    