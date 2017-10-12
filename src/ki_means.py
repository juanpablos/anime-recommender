import random

import numpy as np


class KIMeans:
    centroids = list()
    belongs_to = list()

    def __init__(self, k, max_iter=1000, delta=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.delta = delta

    @staticmethod
    def get_distance(center, point):
        d = 0
        # Only calculate distances for no null dimensions
        for d1, d2 in zip(center, point):
            if d2 != 0:
                d += (d1 - d2) ** 2
        return d ** (1 / 2)

    # Normal implementation of K-means
    def fit(self, data):
        num_instances, dimensionality = data.shape
        self.centroids = data[random.sample(range(0, num_instances), self.k)]
        old_centroids = np.zeros((self.k, dimensionality))
        self.belongs_to = np.zeros((num_instances, 1))
        changed = True
        iterations = 0
        while changed and iterations < self.max_iter:
            iterations += 1
            print(iterations)
            changed = self.centroids_changed(old_centroids)
            for index_instance, instance in enumerate(data):
                dist_vec = np.zeros((self.k, 1))
                for index_centroid, centroid in enumerate(self.centroids):
                    dist_vec[index_centroid] = self.get_distance(centroid, instance)
                self.belongs_to[index_instance, 0] = np.argmin(dist_vec)

            tmp_centroids = np.zeros((self.k, dimensionality))

            for index in range(len(self.centroids)):
                instances_in = [i for i in range(len(self.belongs_to)) if self.belongs_to[i] == index]
                new_centroid = self.mean(data[instances_in], dimensionality)
                tmp_centroids[index, :] = new_centroid
            old_centroids = self.centroids
            self.centroids = tmp_centroids

        return self

    # Checks if the centroids changed more than delta
    def centroids_changed(self, old_centroids):
        for dc1, dc2 in zip(self.centroids, old_centroids):
            diff = dc1 - dc2
            for dim in diff:
                if dim > self.delta:
                    return True
        return False

    def get_correlations(self):
        return [self.centroids, self.belongs_to]

    # Mean of current cluster only considering shared dimensions
    @staticmethod
    def mean(data, dimensions):
        if len(data) != 0:
            index_to_consider = [i + 1 for i in range(dimensions)]
            for data_point in data:
                for i, dim in enumerate(data_point):
                    if dim == 0:
                        index_to_consider[i] = 0
            res = np.zeros(dimensions)
            for p in data:
                for index in index_to_consider:
                    if index != 0:
                        res[index - 1] += p[index - 1]
            res /= len(data)
            return res

        return np.zeros(dimensions)

    def mean_distance(self, center_point, points):
        distance = 0
        if len(points) > 0:
            for p in points:
                distance += self.get_distance(center_point, p)
            return distance / len(points)
        return 0.


if __name__ == '__main__':
    a = np.array([[7, 7, 0, 3],
                  [7, 10, 1, 5],
                  [1, 7, 10, 1],
                  [7, 9, 0, 8],
                  [1, 3, 2, 4],
                  [1, 5, 10, 3],
                  [6, 9, 4, 9],
                  [2, 3, 2, 4],
                  ])
    ki = KIMeans(4, 100, 1e-4)
    ki.fit(a)
    print(ki.get_correlations()[0])
    print(ki.get_correlations()[1])
