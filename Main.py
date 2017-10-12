import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.ki_means import KIMeans


def get_clusters(dataset, belongs_to, n_clusters):
    clusters = list()
    for i in range(n_clusters):
        clusters_index = [index for index in range(len(belongs_to)) if belongs_to[index][0] == float(i)]
        cluster = list()
        for ci in clusters_index:
            cluster.append(dataset[ci])
        clusters.append(cluster)
    return clusters


def get_sse(dataset, max_k, step=1):
    sse = list()
    for k in range(1, max_k, step):
        print("iteration: {}".format(k))
        sse_k = 0
        ki_means = KIMeans(k=k, max_iter=100, delta=1e-1)
        ki_means.fit(dataset)
        centroids, belongs_to = ki_means.get_correlations()
        clusters = get_clusters(dataset, belongs_to, len(centroids))
        for index, c in enumerate(centroids):
            mean = ki_means.mean_distance(c, clusters[index])
            for p in clusters[index]:
                sse_k += (ki_means.get_distance(c, p) - mean) ** 2
        sse.append(sse_k)
    return sse


data = list()
with open('out_file.csv', 'r') as f:
    ff = csv.reader(f)
    for line in ff:
        if line:
            data.append([int(i) for i in line[1:]])

#data2 = np.array(data)

data2 = sparse.coo_matrix(np.array(data))
svd = TruncatedSVD(n_components=200, n_iter=7, random_state=0)
reduced_data = svd.fit_transform(data2)

error = get_sse(reduced_data, 101, step=20)
plt.plot([i for i in range(1, 101, 20)], error, 'o-')
plt.show()


