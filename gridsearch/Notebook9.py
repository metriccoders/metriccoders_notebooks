from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

dataset = load_digits()
clf = RandomForestClassifier()
parameters = {
    "n_estimators": [1, 5, 8, 10],
    "criterion": ["gini", "entropy", "loss"]
}
est = HalvingGridSearchCV(
    clf, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)