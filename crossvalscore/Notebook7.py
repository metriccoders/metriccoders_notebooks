from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


dataset = load_iris()
X = dataset.data
y = dataset.target
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
