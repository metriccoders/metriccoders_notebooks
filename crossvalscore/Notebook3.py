from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier


dataset = load_iris()
X = dataset.data
y = dataset.target
clf = GaussianProcessClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
