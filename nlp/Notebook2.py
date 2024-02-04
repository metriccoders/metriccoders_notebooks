import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#Text classification using scikit-learn
#1. Load libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


#2. load dataset
dataset = fetch_20newsgroups()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#3. convert data into feature vector
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


#4. Train classifier
clf = LogisticRegression()
log = clf.fit(X_train, y_train)
pred = log.predict(X_test)

print(metrics.classification_report(y_test, pred))


