from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

dataset = load_iris()
clf = AdaBoostClassifier()

parameters = {
    "n_estimators": [100, 200, 300],
    "algorithm": ["SAMME", "SAMME.R"]
}

est = GridSearchCV(clf, param_grid=parameters)
est.fit(dataset.data, dataset.target)

print("Best Score:",est.best_score_)
print("Best Params:", est.best_params_)

