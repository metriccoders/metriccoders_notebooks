from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris()
X = dataset.data
y = dataset.target
clf = RandomForestClassifier(
    n_estimators=6
)

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=68)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)

scores = cross_val_score(clf, X, y, cv=3)
print(scores)

cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)