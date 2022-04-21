import csv
import template

file = open("SWD/02_Annotations/ann_score_globalkey.csv")

csvreader = csv.reader(file)
header = next(csvreader)
header = header[0].split(';')

rows = []

for row in csvreader:
    
    newrow = row[0].split(';')

    newrow[1] = newrow[1].split('"')[1].split(':')
    
    if newrow[1][0][-1] == 'b':
        newrow[1] = template.flat2sharp[newrow[1][0]] + ' ' + template.flat2sharp[newrow[1][1]]
    else:
        newrow[1] = newrow[1][0] + ' ' + template.flat2sharp[newrow[1][1]]

    rows.append(newrow)

file.close()
print(rows)
filename = 'SWD/02_Annotations/ann_score_globalkey_2.csv'
with open(filename, 'w', newline="") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(rows) 