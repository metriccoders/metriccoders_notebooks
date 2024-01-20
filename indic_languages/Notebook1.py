from sklearn.feature_extraction.text import CountVectorizer
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


print(data, data_labels)


vectorizer = CountVectorizer(
    analyzer="word",
    lowercase=False
)

features = vectorizer.fit_transform(data)
print(features)

features_nd = features.toarray()

print(features_nd)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_nd, data_labels, test_size=0.2, random_state=50)

from sklearn.linear_model import LogisticRegression

est = LogisticRegression()

est.fit(X_train, y_train)

y_pred = est.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.pipeline import Pipeline

model_pipeline = Pipeline([
    ("classifier", LogisticRegression())
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

print(accuracy_score(y_test, y_pred))
import joblib

joblib.dump(est, "model.joblib")

loaded_model_pipeline = joblib.load("model.joblib")

y_pred_loaded = loaded_model_pipeline.predict(X_test)

assert all(y_pred == y_pred_loaded)
print("Model successfully loaded and predictions matched")