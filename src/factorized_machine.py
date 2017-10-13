import numpy as np
import csv
import ast
import time
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

train = np.copy(data2)
test = np.copy(data2)

for i in range(int(len(data2) * 0.8)):
    for j in range(len(test[i])):
        test[(i,j)] = 0.

for i in range(int(len(data2) * 0.8), int(len(data2))):
    for j in range(len(train[i])):
        train[(i,j)] = 0.

NUM_THREADS = 4
NUM_EPOCHS = 30

#train = data2[:int(len(data2) * 0.8)]
#test = data2[int(len(data2) * 0.8):]
train = sparse.csr_matrix(train)
test = sparse.csr_matrix(test)
item_features = sparse.csr_matrix(item_features)

model = LightFM(loss='warp', learning_rate=0.05)

model.fit(interactions=train, epochs=NUM_EPOCHS, item_features=item_features, num_threads=NUM_THREADS)

predictions = model.predict(0, data2[0]).argsort()[-5:][::-1]

for p in predictions:
    print(anime_id_name[str(p+1)])


train_auc = auc_score(model, train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)

test_auc = auc_score(model, test, train_interactions=train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)

# Set biases to zero
model.item_biases *= 0.0

test_auc = auc_score(model, test, train_interactions=train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)
