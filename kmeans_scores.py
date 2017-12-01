import csv

from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k

from data_utils import fetch_anime

NUM_THREADS = 5
NUM_EPOCHS = 30
NUM_COMPONENTS = 30
ITEM_ALPHA = 1e-6

train_file = '../user_interactions_train.csv'
test_file = '../user_interactions_test.csv'
kmeans_lsa100 = '../kmeans_score0_lsa_dim100_out.csv'
kmeans_lsa1000 = '../kmeans_score0_lsa_dim1000_out.csv'
kmeans = '../kmeans_score0_out.csv'
kimeans = '../kimeans_out.csv'
genres = '../anime_genre_ind.csv'

keys = {'kmeans_lsa100', 'kmeans_lsa1000', 'kmeans', 'kimeans'}
# Genres win

data_klsa_100 = fetch_anime(train_file=train_file,
                            test_file=test_file,
                            user_features_file=kmeans_lsa100,
                            features_file=genres)

data_klsa_1000 = fetch_anime(train_file=train_file,
                             test_file=test_file,
                             user_features_file=kmeans_lsa1000,
                             features_file=genres)

data_kmeans = fetch_anime(train_file=train_file,
                          test_file=test_file,
                          user_features_file=kmeans,
                          features_file=genres)

data_kimeans = fetch_anime(train_file=train_file,
                           test_file=test_file,
                           user_features_file=kimeans,
                           features_file=genres)

data_ksets = {'kmeans_lsa1000': data_klsa_1000,
              'kmeans_lsa100': data_klsa_100,
              'kmeans': data_kmeans,
              'kimeans': data_kimeans}

print('Starting Fit')
model = {}
for k in keys:
    print('Fitting: {}'.format(k))
    model[k] = LightFM(loss='warp', learning_rate=0.05,
                       item_alpha=ITEM_ALPHA)
    model[k].fit(interactions=data_ksets[k]['train_set'],
                 user_features=data_ksets[k]['user_features'],
                 item_features=data_ksets[k]['item_features'],
                 epochs=NUM_EPOCHS,
                 num_threads=NUM_THREADS)

print('Fit Done')
score = {'kmeans_lsa1000': {},
         'kmeans_lsa100': {},
         'kmeans': {},
         'kimeans': {}}

f = open('scores_kmeans_genre.csv', 'w', encoding='utf-8', newline='\n')
o = csv.writer(f)
o.writerow(['model', 'train_score', 'test_score', 'precision_at_k_train', 'precision_at_k_test'])

print('Scoring')
for k in keys:
    print('Scoring: {}'.format(k))
    score[k]['train_set'] = auc_score(model[k], data_ksets[k]['train_set'],
                                      user_features=data_ksets[k]['user_features'],
                                      item_features=data_ksets[k]['item_features'],
                                      num_threads=NUM_THREADS).mean()
    score[k]['test_set'] = auc_score(model[k], data_ksets[k]['test_set'],
                                     train_interactions=data_ksets[k]['train_set'],
                                     user_features=data_ksets[k]['user_features'],
                                     item_features=data_ksets[k]['item_features'],
                                     num_threads=NUM_THREADS).mean()

    score[k]['precision_at_k_train'] = precision_at_k(model[k], data_ksets[k]['train_set'],
                                                      user_features=data_ksets[k]['user_features'],
                                                      item_features=data_ksets[k]['item_features'],
                                                      num_threads=NUM_THREADS).mean()
    score[k]['precision_at_k_test'] = precision_at_k(model[k], data_ksets[k]['test_set'],
                                                     train_interactions=data_ksets[k]['train_set'],
                                                     user_features=data_ksets[k]['user_features'],
                                                     item_features=data_ksets[k]['item_features'],
                                                     num_threads=NUM_THREADS).mean()

    o.writerow([k, score[k]['train_set'], score[k]['test_set'], score[k]['precision_at_k_train'],
                score[k]['precision_at_k_test']])

print('Scoring Done')
f.close()
