from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)