from sklearn.cluster import KMeans
import numpy as np
X = np.array([
    [1, 10], [2,7], [6,5],
    [10, 2], [4,7], [7,8]
])
kmeans = KMeans(n_clusters=2, random_state=67, n_init="auto").fit(X)

print(kmeans.predict([[2,3],[4,8]] ))
print(kmeans.cluster_centers_)