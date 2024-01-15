from sklearn.datasets import load_digits
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

dataset = load_digits()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(
    criterion="gini"
)
scoring = ["precision_macro", "recall_macro"]
scores = cross_validate(clf, X, y, scoring=scoring)
keys = scores.keys()
print(keys)
for x in keys:
    print("{0}: {1}", x, scores[x])


