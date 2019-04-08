import nltk
import random   #shuffle the dataset from highly ordered to unordered
from nltk.corpus import movie_reviews

#go to text_classification_sentiment_movie reviews.py
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents) 

#go to text_classification_sentiment_movie reviews.py
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#go to text_classification_sentiment_movie reviews.py
all_words =nltk.FreqDist(all_words)

#go to text_classification_sentiment_movie reviews.py
word_features = list(all_words.keys())[:3000]

#new
#finding features that we will be using
def find_features(document):
    words = set(document)    #every single word of the document not their count 
    features = {}
    for w in word_features:     #w in 3000most popular words
        features[w] = (w in words)   #return true in features[w] if w exists in word_features list of top 3000 words 

    return features

#print the result
print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]   #return a list of the top 3000 words with the true or false depending on weathr they exists in the document or not
