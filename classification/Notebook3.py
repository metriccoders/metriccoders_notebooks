from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier

iris = load_iris()
X = iris.data
y = iris.target
print("Data:", X[:10])
print("Target:", y[:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = GaussianProcessClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))