import csv
import template
from os import listdir

versions = ['FI66', 'FI80', 'HU33', 'SC06']
localkey_folder = "SWD/02_Annotations/ann_audio_localkey-ann1"
localkey_files = listdir(localkey_folder)

for file in localkey_files:
    
    rows = []
    version = file.split('_')[2]
    if version not in [v + '.csv' for v in versions]:
        continue
    version = version.split('.')[0]

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


    with open('{}-{}/{}'.format(localkey_folder, version, file), 'w', newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(rows)
