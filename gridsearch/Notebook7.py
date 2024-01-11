from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.linear_model import RidgeClassifier
import numpy as np

dataset = load_iris()
clf = RidgeClassifier()
parameters = {
    "alpha": [1.0, 2.0, 3.0, 4.0],
    "fit_intercept": [True, False]
}
est = HalvingGridSearchCV(
    clf, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)