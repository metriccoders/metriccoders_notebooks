from sklearn.datasets import load_breast_cancer
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
clf = AdaBoostClassifier(
    n_estimators=4
)
scoring = ["precision_macro", "recall_macro"]
scores = cross_validate(clf, X, y, scoring=scoring)
keys = scores.keys()
print(keys)
for x in keys:
    print("{0}: {1}", x, scores[x])


