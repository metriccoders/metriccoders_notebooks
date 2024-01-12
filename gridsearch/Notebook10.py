from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

dataset = load_digits()
clf = AdaBoostClassifier()
parameters = {
    "n_estimators": [1, 5, 8, 10, 50, 100],
    "algorithm": ["SAMME", "SAMME.R"]
}
est = HalvingGridSearchCV(
    clf, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)