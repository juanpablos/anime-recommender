import ast
import csv
import itertools

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from ki_means import KIMeans


def parse(data):
    for line in data:
        if not line:
            continue
        user_id = line[0]
        for anime in line[1:]:
            anime_norm_id, anime_score = ast.literal_eval(anime)
            yield int(user_id) - 1, int(anime_norm_id) - 1, int(anime_score)


def get_dimensions(train_data, test_data):
    print("Getting dimensions")
    user_ids = set()
    anime_ids = set()
    for uid, aid, _ in itertools.chain(train_data, test_data):
        user_ids.add(uid)
        anime_ids.add(aid)
    rows = max(user_ids) + 1
    cols = max(anime_ids) + 1

    return rows, cols


def build_interaction_matrix(rows, cols, data, min_rating):
    print("Building interaction matrix")
    matrix = sp.lil_matrix((rows, cols), dtype=np.int32)
    for user_id, anime_id, score in data:
        if score >= min_rating:
            matrix[user_id, anime_id] = score

    return matrix.tocoo()


def build_item_metadata_matrix(num_items, metadata):
    print("Building metadata matrix")
    header = next(metadata)
    n_items, n_features = [int(i) for i in header]
    assert num_items == n_items

    id_features = sp.identity(num_items, format='csr', dtype=np.float32)
    features = sp.lil_matrix((num_items, n_features), dtype=np.float32)

    for line in metadata:
        # blank line
        if not line:
            continue
        # no feature
        if len(line) < 2:
            continue
        anime_id = int(line[0]) - 1

        for feature in line[1:]:
            features[anime_id, int(feature) - 1] = 1.0

    return id_features, features.tocsr()


def get_user_clustering_dimensions(data):
    print("Getting clustering dimensions")
    user_ids = set()
    clusters = set()
    for uid, _, c in data:
        user_ids.add(int(uid))
        clusters.add(int(c))
    rows = max(user_ids)
    cols = max(clusters) + 1

    return rows, cols


def build_user_feature_matrix(num_users, user_data, clusters):
    print("Building user feature matrix")

    id_features = sp.identity(num_users, format='csr', dtype=np.float32)
    features = sp.lil_matrix((num_users, clusters), dtype=np.float32)
    for line in user_data:
        # blank line
        if not line:
            continue
        # no feature
        if len(line) < 3:
            continue
        user_id = int(line[0]) - 1

        features[user_id, int(line[2])] = 1.0

    return id_features, features.tocsr()


def fetch_anime(train_file, test_file, user_features_file=None, features_file=None, min_score=0):
    print("Start Fetching")
    user_feature_matrix = None
    item_feature_matrix = None
    with open(train_file, 'r', encoding='utf-8') as train_f:
        with open(test_file, 'r', encoding='utf-8') as tests_f:
            train = csv.reader(train_f, delimiter=',')
            test = csv.reader(tests_f, delimiter=',')
            num_users, num_items = get_dimensions(parse(train), parse(test))
            train_f.seek(0)
            tests_f.seek(0)
            train_matrix = build_interaction_matrix(num_users, num_items, parse(train), min_score)
            test_matrix = build_interaction_matrix(num_users, num_items, parse(test), min_score)
            assert train_matrix.shape == test_matrix.shape

            if user_features_file:
                with open(user_features_file, 'r', encoding='utf-8') as user_features_f:
                    features = csv.reader(user_features_f, delimiter=',')
                    n_users, clusters = get_user_clustering_dimensions(features)
                    assert num_users == n_users
                    user_features_f.seek(0)
                    id_features, feature_matrix = build_user_feature_matrix(num_users, features, clusters)
                    user_feature_matrix = sp.hstack([id_features, feature_matrix]).tocsr()

            if features_file:
                with open(features_file, 'r', encoding='utf-8') as item_features_f:
                    features = csv.reader(item_features_f, delimiter=',')
                    id_features, feature_matrix = build_item_metadata_matrix(num_items, features)
                    item_feature_matrix = sp.hstack([id_features, feature_matrix]).tocsr()

    return {'train_set': train_matrix,
            'test_set': test_matrix,
            'user_features': user_feature_matrix,
            'item_features': item_feature_matrix}


def fetch_all_data(data_file, min_score):
    print("Reading interaction file")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = csv.reader(f, delimiter=',')
        num_users, num_items = get_dimensions(parse(data), [])
        f.seek(0)
        print("Building matrix...")
        interaction_matrix = build_interaction_matrix(num_users, num_items, parse(data), min_score)
        return interaction_matrix


def run_kmeans(data, k, user_ids_file, out_file, debug=False):
    print("Starting K means")
    if debug:
        labeler = KMeans(n_clusters=k, random_state=0)
    else:
        labeler = KMeans(n_clusters=k)
    print("Fitting...")
    labeler.fit(data.tocsr())

    with open(user_ids_file, 'r', encoding='utf-8') as f:
        with open(out_file, 'w', encoding='utf-8', newline="\n") as out:
            reader = csv.reader(f)
            writer = csv.writer(out)
            for n_line, line in enumerate(reader):
                writer.writerow(line + [labeler.labels_[n_line]])


def run_lsa_kmeans(data, k, user_ids_file, out_file, reduce_dim_to=1000, iter=10, debug=False):
    print("Starting LSA")
    if debug:
        svd = TruncatedSVD(n_components=reduce_dim_to, n_iter=iter, random_state=0)
    else:
        svd = TruncatedSVD(n_components=reduce_dim_to, n_iter=iter)
    print("Fitting LSA...")
    reduced_data = svd.fit_transform(data)
    run_kmeans(sp.coo_matrix(reduced_data), k, user_ids_file, out_file, debug)


def run_kimeans(data, k, iter, user_ids_file, out_file, debug=False):
    print("Starting Ki means")
    ki = KIMeans(k, iter)
    print("Fitting...")
    ki.fit(np.array(data.todense()))
    clusters = ki.get_correlations()[1]
    with open(user_ids_file, 'r', encoding='utf-8') as f:
        with open(out_file, 'w', encoding='utf-8', newline="\n") as out:
            reader = csv.reader(f)
            writer = csv.writer(out)
            for n_line, line in enumerate(reader):
                writer.writerow(line + [int(clusters[n_line][0])])


if __name__ == "__main__":
    dicta = fetch_anime(train_file='user_interactions_train.csv', test_file='user_interactions_test.csv',
                       user_features_file='kmeans_score0_out.csv', features_file='anime_director_ind.csv', min_score=0)
    print(dicta)
    # anime_data = fetch_all_data('user_interactions.csv', 0)
    # run_lsa_kmeans(anime_data, 50, user_ids_file='Data/users_id.csv', out_file='kmeans_lsa_dim100_out.csv', debug=True, reduce_dim_to=100)
    # run_kimeans(anime_data, 50, iter=100, user_ids_file='Data/users_id.csv', out_file='kimeans_out.csv', debug=True)
