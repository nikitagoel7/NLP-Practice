#how accurate are we calculating the votes-----accuracy on positive and accuracy on negative documents
import nltk
import random   #shuffle the dataset from highly ordered to unordered
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier  #wrapper to include scikit algorithms within nltk 

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode 

#go to combining_algo_with_vote.py
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers :
            v = c.classify(features)  #for every classifier finding votes
            votes.append(v)
        return mode(votes)   #who got the most votes

    def confidence(self, features):
        votes = []
        for c in self._classifiers :
            v = c.classify(features) 
            votes.append(v)

        choice_votes = votes.count(mode(votes))  #count how many number of occurences of max votes were there in that list
        conf = choice_votes / len(votes)
        return conf
     

#go to text_classification_sentiment_movie reviews.py
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#new
#random.shuffle(documents) ---- commented to make first 1000 positive and next 1000 negative

#go to text_classification_sentiment_movie reviews.py
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#go to text_classification_sentiment_movie reviews.py
all_words =nltk.FreqDist(all_words)

#go to text_classification_sentiment_movie reviews.py
word_features = list(all_words.keys())[:3000]

#go to word_features.py
def find_features(document):
    words = set(document)    
    features = {}
    for w in word_features:     
        features[w] = (w in words)   

    return features
featuresets = [(find_features(rev), category) for (rev, category) in documents]   #return a dict of the top 3000 words with the true or false depending on weather they exists in the document or not


#go to naive_bayes.py
"""
Basic algorithm which can be scaled and is not complex in application
posterior = prior occurences x likelihood / evidence
Really easy to understand
Uses nltk
It is not that accurate and reliable and gives different results on different runs
"""
#new
#positive data example:
training_set = featuresets[:1900]
test_set = featuresets[1900:]

#negative data example: ---- changing the training_set and test_set for negative reviews
training_set = featuresets[100:]
test_set = featuresets[:100]

#go to save_classifier_with_pickle.py
classifier_f = open ("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print ("Orignal Naive Bayes Also accuracy: ", (nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(15)   #go to save_classifier_with_pickle.py


# go nltk_scikit_incorporation.py
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print ("MNB_classifier Also accuracy: ", (nltk.classify.accuracy(MNB_classifier, test_set))*100) 

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print ("BernoulliNB_classifier Also accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100) 

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("LogisticRegression_classifier Also accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100) 

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print ("SGDClassifier_classifier Also accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100) 

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("LinearSVC_classifier Also accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100) 

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print ("NuSVC_classifier Also accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100) 

#go to combining_algo_with_vote.py
voted_classifier = VoteClassifier(classifier, NuSVC_classifier, LinearSVC_classifier,SGDClassifier_classifier,LogisticRegression_classifier,MNB_classifier, BernoulliNB_classifier)
print ("voted_classifier accuracy percent", (nltk.classify.accuracy(voted_classifier, test_set))*100)

##print ("Classification:", voted_classifier.classify(test_set[0][0]), "Confidence: %", voted_classifier.confidence(test_set[0][0]))
##print ("Classification:", voted_classifier.classify(test_set[1][0]), "Confidence: %", voted_classifier.confidence(test_set[1][0]))
##print ("Classification:", voted_classifier.classify(test_set[2][0]), "Confidence: %", voted_classifier.confidence(test_set[2][0]))
##print ("Classification:", voted_classifier.classify(test_set[3][0]), "Confidence: %", voted_classifier.confidence(test_set[3][0]))
##print ("Classification:", voted_classifier.classify(test_set[4][0]), "Confidence: %", voted_classifier.confidence(test_set[4][0]))
##print ("Classification:", voted_classifier.classify(test_set[5][0]), "Confidence: %", voted_classifier.confidence(test_set[5][0]))
