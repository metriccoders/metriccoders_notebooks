from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.linear_model import LassoLars
import numpy as np

dataset = load_diabetes()
regr = LassoLars()
parameters = {
    "alpha": [0.1, 0.2, 0.3],
    "fit_intercept": [True, False]
}
est = HalvingGridSearchCV(
    regr, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)