
import csv
anime_id_name = {}
with open('general.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        anime_id_name[line[0]] = line[1]