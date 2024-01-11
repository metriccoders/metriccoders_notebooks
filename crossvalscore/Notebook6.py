from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier


dataset = load_iris()
X = dataset.data
y = dataset.target
clf = MLPClassifier(max_iter=5000)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
