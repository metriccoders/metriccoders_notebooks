from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.linear_model import ElasticNet
import numpy as np

dataset = load_diabetes()
regr = ElasticNet()
parameters = {
    "alpha": [0.1, 0.2, 0.3],
    "l1_ratio": [0.5, 0.6, 0.7]
}
est = HalvingGridSearchCV(
    regr, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)