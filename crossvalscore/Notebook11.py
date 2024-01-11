from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB


dataset = load_digits()
X = dataset.data
y = dataset.target
clf = GaussianNB()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
