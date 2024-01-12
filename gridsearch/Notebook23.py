from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.linear_model import SGDClassifier
import numpy as np

dataset = load_breast_cancer()
clf = SGDClassifier()
parameters = {
    "loss": ["hinge", "log_loss", "modified_huber",
             "squared_hinge", "perceptron", "squared_error",
             "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    "penalty": ["l2", "l1", "elasticnet", None]
}
est = HalvingGridSearchCV(
    clf, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)