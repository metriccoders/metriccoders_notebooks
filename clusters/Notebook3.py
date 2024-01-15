from sklearn.cluster import KMeans
import numpy as np

X = np.array([[2,9], [1,8], [3,7],
              [4,5], [6,7], [8,9]])
kmeans = KMeans(n_clusters=3, n_init="auto", random_state=89).fit(X)
print(kmeans.predict([[1,6], [1,8]]))
print(kmeans.cluster_centers_)