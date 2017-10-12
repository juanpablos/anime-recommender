import numpy as np
import csv
import ast
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score


anime_id_name = {}
new_id_dict = {}

with open('new_anime_id.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        new_id_dict[line[0]] = line[1]

with open('general.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
    	if line[0] == 'anime_id':
    		continue
    	anime_id_name[new_id_dict[line[0]]] = line[1]

item_features = np.zeros((13510, 44))

with open('anime_dict.csv', 'r') as fm:
	ff = csv.reader(fm)
	for line in ff:
		a_id = int(line[0])
		a_genres = ast.literal_eval(line[1])
		for g in a_genres:
			item_features[(a_id-1, g-1)] = 1


data = list()
with open('out_file.csv', 'r') as f:
    ff = csv.reader(f)
    for line in ff:
        if line:
            data.append([int(i) for i in line[2:]])

data2 = np.array(data)


NUM_THREADS = 4
NUM_EPOCHS = 30


train = data2[:int(len(data2) * 0.8)]
test = data2[int(len(data2) * 0.8):]
train = sparse.csr_matrix(train)
test = sparse.csr_matrix(test)
item_features = sparse.csr_matrix(item_features)

model = LightFM(loss='warp', learning_rate=0.05)

model.fit(interactions=train, item_features=item_features, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)


predictions = model.predict(0, data2[0]).argsort()[-5:][::-1]
print("Recomendaciones:")
for p in predictions:
	print(anime_id_name[str(p+1)])


