from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

dataset = load_breast_cancer()
clf = DecisionTreeClassifier()
parameters = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"]
}
est = HalvingGridSearchCV(
    clf, param_grid=parameters,
    random_state=np.random.RandomState(0)
)

est.fit(dataset.data, dataset.target)

print("Best Params:", est.best_params_)
print("Best Score:", est.best_score_)