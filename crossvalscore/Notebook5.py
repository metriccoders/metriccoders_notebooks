from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


dataset = load_iris()
X = dataset.data
y = dataset.target
clf = KNeighborsClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
