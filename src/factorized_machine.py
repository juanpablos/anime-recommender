import numpy as np
import csv
import ast
import time
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score

anime_id_name = {}
new_id_dict = {}
users_id = {}

with open('users_id.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        users_id[int(line[0]) - 1] = line[1]

with open('new_anime_id.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        new_id_dict[line[0]] = line[1]

with open('general.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        if line[0] != 'anime_id' and not anime_id_name.get(int(new_id_dict[line[0]])):
            anime_id_name[int(new_id_dict[line[0]])] = line[1]

data = list()
with open('out_file.csv', 'r') as f:
    ff = csv.reader(f)
    for line in ff:
        if line:
            data.append([int(i) for i in line[1:]])

data2 = np.array(data)

item_features = np.zeros((len(data2[0]), 44))

with open('anime_dict.csv', 'r') as fm:
    ff = csv.reader(fm)
    for line in ff:
        a_id = int(line[0])
        a_genres = ast.literal_eval(line[1])
        for g in a_genres:
            item_features[(a_id - 1, g - 1)] = 1

train = np.copy(data2)
test = np.copy(data2)

for i in range(int(len(data2) * 0.8)):
    for j in range(len(test[i])):
        test[(i, j)] = 0.

for i in range(int(len(data2) * 0.8), int(len(data2))):
    for j in range(len(train[i])):
        train[(i, j)] = 0.

NUM_THREADS = 4
NUM_EPOCHS = 30
NUM_COMPONENTS = 30
ITEM_ALPHA = 1e-6

train = sparse.csr_matrix(train)
test = sparse.csr_matrix(test)
item_features = sparse.csr_matrix(item_features)

model = LightFM(loss='warp', learning_rate=0.05)

model.fit(interactions=train, epochs=NUM_EPOCHS, item_features=item_features, num_threads=NUM_THREADS)

prediction_id = 0

predictions = model.predict(0, data2[prediction_id]).argsort()[-20:][::-1]

print("Predictions for user {}".format(users_id[prediction_id]))
for p in predictions:
    print(anime_id_name[p + 1])

train_auc = auc_score(model, train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)

test_auc = auc_score(model, test, train_interactions=train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)

# Set biases to zero
model.item_biases *= 0.0

test_auc = auc_score(model, test, train_interactions=train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)
