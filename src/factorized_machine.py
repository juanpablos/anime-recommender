import numpy as np
import csv
import ast
from scipy import sparse
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k


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

data = sparse.csr_matrix(data2[500:])
item_features = sparse.csr_matrix(item_features)

model = LightFM(loss='warp')
model.fit(interactions=data, epochs=30, num_threads=2, item_features=item_features)
print("Prediccion")
print(model.predict(0, data2[0]).argsort()[-5:][::-1])