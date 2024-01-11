from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier


dataset = load_wine()
X = dataset.data
y = dataset.target
clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
