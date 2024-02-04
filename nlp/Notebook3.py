# Different types of vectorization

sentences = [
    "This is awesome",
    "Dravid is the best cricketer",
    "Test Cricket is the ultimate"
]

#1. Bag of Words
#Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)
print(X.toarray())


#2. TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)
print(X.toarray())

#3. Continuous Bag of Words
from gensim import models
custom_word2vec_model = models.Word2Vec(sentences=sentences, min_count=1)
print(custom_word2vec_model)





