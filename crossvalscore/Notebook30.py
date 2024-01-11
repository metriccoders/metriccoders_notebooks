from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet


dataset = load_diabetes()
X, y = dataset.data, dataset.target

est = ElasticNet()

scores = cross_val_score(est, X, y, cv=5)

print(scores)
