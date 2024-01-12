from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

dataset = load_diabetes()
regr = HistGradientBoostingRegressor()
parameters = {
    "loss": ["squared_error", "absolute_error", "huber", "quantile"],
    "learning_rate": [0.1, 0.2, 0.3]
}
est = HalvingGridSearchCV(
    regr, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)