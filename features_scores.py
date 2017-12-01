import numpy as np
import csv
import ast
import time
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k
from MainClusters import fetch_anime


NUM_THREADS = 2
NUM_EPOCHS = 30
ITEM_ALPHA = 1e-6

train_file = '../user_interactions_train.csv'
test_file = '../user_interactions_test.csv'
kmeans_lsa100 = '../kmeans_score0_lsa_dim100_out.csv'
kmeans_lsa1000 = '../kmeans_score0_lsa_dim1000_out.csv'
kmeans = '../kmeans_score0_out.csv'
kimeans = '../kimeans_out.csv'
directors = '../anime_director_ind.csv'
genres = '../anime_genre_ind.csv'
studios = '../anime_studios_ind.csv'


data_basic = fetch_anime(train_file=train_file, test_file=test_file)

data_genre = fetch_anime(train_file=train_file,
 test_file=test_file, features_file=genres)

data_directors = fetch_anime(train_file=train_file,
 test_file=test_file, features_file=directors)

data_studios = fetch_anime(train_file=train_file,
 test_file=test_file, features_file=studios)


data_sets = {'basic':data_basic, 'genre':data_genre,
 'directors':data_directors, 'studios':data_studios}


print('Starting Fit')
model = {}
keys = ['basic', 'genre', 'directors', 'studios']
for k in keys:
    print('Fitting: {}'.format(k))
    model[k] = LightFM(loss='warp',
        learning_rate=0.05,
        item_alpha=ITEM_ALPHA)
    if k == 'basic':
        model[k].fit(interactions=data_sets[k]['train_set'],
            epochs=NUM_EPOCHS,
            num_threads=NUM_THREADS)
    else:
        model[k].fit(interactions=data_sets[k]['train_set'],
            item_features=data_sets[k]['item_features'],
            epochs=NUM_EPOCHS,
            num_threads=NUM_THREADS)

print('Fit Done')
score = {'basic':{},'genre':{},'directors':{},'studios':{}}

f = open('model_scores_warp.csv', 'w', encoding='utf-8', newline='\n')
o = csv.writer(f)
o.writerow(['model','train_score','test_score', 'precision_at_k_train', 'precision_at_k_test'])
max_k = ''
max_score = -1
print('Scoring')
for k in keys:
    print('Scoring: {}'.format(k))
    if k == 'basic':
        score[k]['train_set'] = auc_score(model[k], data_sets[k]['train_set'],
         num_threads=NUM_THREADS).mean()
        score[k]['test_set'] = auc_score(model[k], data_sets[k]['test_set'],
            train_interactions=data_sets[k]['train_set'],
            num_threads=NUM_THREADS).mean()
        score[k]['precision_at_k_train'] = precision_at_k(model[k], data_sets[k]['train_set'],
            num_threads=NUM_THREADS).mean()
        score[k]['precision_at_k_test'] = precision_at_k(model[k], data_sets[k]['test_set'],
            train_interactions=data_sets[k]['train_set'],
            num_threads=NUM_THREADS).mean()
    else:
        score[k]['train_set'] = auc_score(model[k], data_sets[k]['train_set'],
            item_features=data_sets[k]['item_features'],
            num_threads=NUM_THREADS).mean()
        score[k]['test_set'] = auc_score(model[k], data_sets[k]['test_set'],
            train_interactions=data_sets[k]['train_set'],
            item_features=data_sets[k]['item_features'],
            num_threads=NUM_THREADS).mean()
        score[k]['precision_at_k_train'] = precision_at_k(model[k], data_sets[k]['train_set'], 
            item_features=data_sets[k]['item_features'],
            num_threads=NUM_THREADS).mean()
        score[k]['precision_at_k_test'] = precision_at_k(model[k], data_sets[k]['test_set'],
            train_interactions=data_sets[k]['train_set'], 
            item_features=data_sets[k]['item_features'],
            num_threads=NUM_THREADS).mean()

    if k != 'basic' and score[k]['test_set'] > max_score:
        max_score = score[k]['test_set']
        max_k = k

    o.writerow([k, score[k]['train_set'], score[k]['test_set'], score[k]['precision_at_k_train'], score[k]['precision_at_k_test']])

print('Scoring Done')
f.close()

print(max_k)
