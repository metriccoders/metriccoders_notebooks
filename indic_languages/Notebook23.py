from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import joblib


data = []
data_labels = []
with open("./datasets/kannada/pos.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('pos')

with open("./datasets/kannada/neg.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('neg')


vectorizer = CountVectorizer(
    analyzer="word",
    lowercase=False
)
features = vectorizer.fit_transform(data)

features_nd = features.toarray()

X_train, X_test, y_train, y_test = train_test_split(features_nd,
                                                    data_labels,
                                                    test_size=0.3,
                                                    random_state=42)

model_pipeline = Pipeline([
    ("clf", Perceptron())
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

print(accuracy_score(y_test, y_pred))

joblib.dump(model_pipeline, "model.joblib")
