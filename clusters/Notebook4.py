import numpy as np
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs


centers = [[1,1], [1,-1], [-1, -1], [-1, 1]]
X, labels_true = make_blobs(
    n_samples=500, centers=centers, cluster_std=0.8, random_state=42
)

af = AffinityPropagation(preference=60, random_state=89).fit(X)

cluster_center_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_center_indices)
print("Estimated number of clusters: %d" % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print(
    "Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(X, labels, metric="sqeuclidean")
)