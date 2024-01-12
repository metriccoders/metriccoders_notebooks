from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.kernel_ridge import KernelRidge
import numpy as np

dataset = load_diabetes()
regr = KernelRidge()
parameters = {
    "gamma": [None, 0.1, 0.2, 0.3],
    "degree": [1,2,3, 4]
}
est = HalvingGridSearchCV(
    regr, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)