from sklearn.datasets import load_wine
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

dataset = load_wine()
clf = KNeighborsClassifier()
parameters = {
    "n_neighbors": [5, 10, 15, 20, 25,30, 35],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", ""]
}
est = HalvingGridSearchCV(
    clf, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)