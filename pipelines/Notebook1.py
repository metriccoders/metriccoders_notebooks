from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=45)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=56)
pipe = Pipeline([
    ("scaler", StandardScaler()),
("clf", ExtraTreeClassifier())
])
print(pipe.fit(X_train, y_train).score(X_test, y_test))