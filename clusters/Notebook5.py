from sklearn.cluster import MeanShift
import numpy as np

X = np.array([[1,1], [2,3], [4,5],
              [1,2], [2,1], [3,2]])

clustering = MeanShift(bandwidth=2).fit(X)
print(clustering.predict([[1,1], [2,0]]))