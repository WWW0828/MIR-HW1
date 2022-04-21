import csv
import template

versions = ['FI66', 'FI80', 'HU33', 'SC06']

file = open("SWD/02_Annotations/ann_audio_globalkey.csv")

csvreader = csv.reader(file)
header = next(csvreader)
header = header[0].split(';')

rows_FI66 = []
rows_FI80 = []
rows_HU33 = []
rows_SC06 = []

for row in csvreader:
    
    newrow = row[0].split(';')

    newrow[1] = newrow[1].split('"')[1]
    if newrow[1] not in versions:
        continue
    
    newrow[2] = newrow[2].split('"')[1].split(':')
    if newrow[2][0][-1] == 'b':
        newrow[2] = template.flat2sharp[newrow[2][0]] + ' ' + template.flat2sharp[newrow[2][1]]
    else:
        newrow[2] = newrow[2][0] + ' ' + template.flat2sharp[newrow[2][1]]

    if newrow[1] == 'FI66':
        rows_FI66.append(newrow)
    elif newrow[1] == 'FI80':
        rows_FI80.append(newrow)
    elif newrow[1] == 'HU33':
        rows_HU33.append(newrow)
    elif newrow[1] == 'SC06':
        rows_SC06.append(newrow)

file.close()
rows = [rows_FI66, rows_FI80, rows_HU33, rows_SC06]

for v_id,v in enumerate(versions):

    filename = 'SWD/02_Annotations/ann_audio_globalkey_' + v + '.csv'
    with open(filename, 'w', newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(rows[v_id])