from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier


dataset = load_iris()
X = dataset.data
y = dataset.target
clf = AdaBoostClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
