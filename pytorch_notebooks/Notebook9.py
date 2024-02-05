# Pytorch text classification using Bag of Words Classifier

#load libraries
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#load dataset
dataset = fetch_20newsgroups( subset="all", remove=("headers", "footers", "quotes"))

#tokenize data
tokenizer = get_tokenizer("basic_english")
tokens = [tokenizer(text) for text in dataset.data]

#build vocabulary
vocab = build_vocab_from_iterator(tokens, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


#convert tokens to integer indices
data_indices = [ [vocab[token]for token in text] for text in tokens]

#create BoW vectors
def create_bow_vector(indices):
    vector = torch.zeros(len(vocab))
    for idx in indices:
        vector[idx] += 1
    return vector


X = torch.stack([ create_bow_vector(indices=indices) for indices in data_indices])
y = torch.tensor(dataset.target)

#create model class
class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(BagOfWordsClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_classes)
        
    def forward(self, x):
        return self.linear(x)


#define loss and optimizer
learning_rate = 0.01
n_epochs = 100

input_size = len(vocab)
output_size = len(dataset.target_names)

model = BagOfWordsClassifier(input_size, output_size)


loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



#training loop
for epoch in range(n_epochs):
    #foward pass to model class
    predicted = model(X)
    #loss
    l = loss(predicted, y)
    #calculate gradients = backward pass
    l.backward()
    #update weights
    optimizer.step()
    #zero gradient
    optimizer.zero_grad()
    print(f"Epoch: {epoch+1}, Loss: {l.item()}")    


