from sklearn.datasets import load_wine
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
import numpy as np

dataset = load_wine()
svc = SVC()
parameters = {
    "kernel": ("linear", "rbf"),
    "C": [1, 10]
}
est = HalvingGridSearchCV(
    svc, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)