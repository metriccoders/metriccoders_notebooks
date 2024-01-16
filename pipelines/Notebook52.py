from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import OrthogonalMatchingPursuitCV


X, y = make_regression(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=56, test_size=0.2)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", OrthogonalMatchingPursuitCV())
])

scores = pipe.fit(X_train, y_train).score(X_test, y_test)
print(scores)