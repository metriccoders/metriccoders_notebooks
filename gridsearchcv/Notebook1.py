from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris()
clf = RandomForestClassifier()

parameters = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"]
}

est = GridSearchCV(clf, param_grid=parameters)
est.fit(dataset.data, dataset.target)

print("Best Score:",est.best_score_)
print("Best Params:", est.best_params_)

