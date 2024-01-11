from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier


dataset = load_digits()
X = dataset.data
y = dataset.target
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
