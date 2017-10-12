from sklearn.decomposition import TruncatedSVD
import csv
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

data = list()
with open('out_file.csv', 'r') as f:
    ff = csv.reader(f)
    for line in ff:
        if line:
            data.append([int(i) for i in line[1:]])

data2 = sparse.coo_matrix(np.array(data))
#svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=0)
#reduced_data = svd.fit_transform(data2)
print("asdasdas")


from sklearn.cluster import KMeans


# 25 works fine
sse = []
for k in range(1, 1001, 50):
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data2.tocsr())
    sse.append(kmeans.inertia_)

plt.plot([i for i in range(1, 1001, 50)], sse, 'o-')
plt.show()