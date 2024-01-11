from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier


dataset = load_digits()
X = dataset.data
y = dataset.target
clf = AdaBoostClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
