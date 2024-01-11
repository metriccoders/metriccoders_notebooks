from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier


dataset = load_digits()
X = dataset.data
y = dataset.target
clf = KNeighborsClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
