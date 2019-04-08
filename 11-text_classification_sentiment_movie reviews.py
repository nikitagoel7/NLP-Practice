import nltk
import random   #shuffle the dataset from highly ordered to unordered
from nltk.corpus import movie_reviews

#new
##documents = [(list(movie_reviews.words(fileid)), category)
##             for category in movie_reviews.categories()
##             for fileid in movie_reviews.fileids(category)]

#building features from words by puting the words from both the categories and then shuffling the documents
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents) #for bringing variatio in the training and testing of data to remove irrelavncy

#print (documents[0])

#all the words in the documents inclusing punctuations
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#converting list to frquency distribution
all_words =nltk.FreqDist(all_words)
#print (all_words.most_common(15))

#print (all_words["common"])  ----> how many times "common" appeared



