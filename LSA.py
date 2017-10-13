import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

data = list()
with open('out_file.csv', 'r') as f:
    ff = csv.reader(f)
    for line in ff:
        if line:
            data.append([int(i) for i in line[1:]])

data2 = sparse.coo_matrix(np.array(data))
svd = TruncatedSVD(n_components=1000, n_iter=10, random_state=0)
reduced_data = svd.fit_transform(data2)
print("__________________")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=25, random_state=0).fit(reduced_data)  # reduced_data.tocsr())
print(kmeans.labels_)

# with open('users_id.csv', 'r', encoding='utf-8') as f:
#     with open('user_id_cluster_lsa1.csv', 'w', encoding='utf-8', newline="\n") as out:
#         writer = csv.writer(out)
#         reader = csv.reader(f)
#
#         for n_line, line in enumerate(reader):
#             writer.writerow(line + [kmeans.labels_[n_line]])

# 25 works fine
sse = []
for k in range(1, 501, 50):
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(reduced_data)
    sse.append(kmeans.inertia_)

plt.plot([i for i in range(1, 501, 50)], sse, 'o-')
plt.show()
