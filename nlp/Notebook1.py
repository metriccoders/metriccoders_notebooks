import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#Text classification using scikit-learn
#1. Load libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#2. Load dataset
dataset = fetch_20newsgroups()

#print(dataset)
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)

print(y)


#3. Convert dataset into feature vector
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)
print(X_train)

X_test = vectorizer.transform(X_test)
print(X_test)

#4. Train classifier

clf = LogisticRegression()
log = clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(metrics.classification_report(y_test, pred))


