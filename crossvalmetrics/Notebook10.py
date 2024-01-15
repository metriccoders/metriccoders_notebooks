from sklearn.datasets import load_digits
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

dataset = load_digits()
X, y = dataset.data, dataset.target
clf = RandomForestClassifier(
    n_estimators=4
)
scoring = ["precision_macro", "recall_macro"]
scores = cross_validate(clf, X, y, scoring=scoring)
keys = scores.keys()
print(keys)
for x in keys:
    print("{0}: {1}", x, scores[x])


