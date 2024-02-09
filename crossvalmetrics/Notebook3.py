from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

dataset = load_iris()
X, y = dataset.data, dataset.target
print("Data:", X)
print("Target:", y)

clf = AdaBoostClassifier(
    n_estimators=6
)

cv = ShuffleSplit(
    n_splits=4, test_size=0.2, random_state=65
)

scores = cross_val_score(clf, X, y, cv = cv)
print(scores)

scores = cross_val_score(clf, X, y, cv=4)
print(scores)