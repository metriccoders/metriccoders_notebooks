from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn import svm

dataset = load_iris()
X, y = dataset.data, dataset.target
clf = svm.SVC(kernel="linear", C=1, random_state=67)
scores = cross_val_score(clf,X, y, cv=5)
print(scores)

scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
print(scores)
from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
print(n_samples)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)