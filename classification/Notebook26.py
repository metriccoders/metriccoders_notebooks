from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

estimator = []
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = SVC()

estimator.append(("clf1", clf1))
estimator.append(("clf2", clf2))
estimator.append(("clf3", clf3))

clf = VotingClassifier(estimators=estimator, voting="hard")
clf.fit(X_train, y_train)

scores = clf.score(X_test, y_test)
print(scores)