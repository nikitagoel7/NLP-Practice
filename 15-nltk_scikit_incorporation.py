#scikit library incorporation
import nltk
import random   #shuffle the dataset from highly ordered to unordered
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier  #wrapper to include scikit algorithms within nltk 

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents) #for bringing variatio in the training and testing of data to remove irrelavncy

#print (documents[0])

#all the words in the documents including punctuations
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#converting list to frquency distribution sorted in the most to least common word
all_words =nltk.FreqDist(all_words)
#print (all_words.most_common(15))

#print (all_words["common"])  ----> how many times "common" appeared


#putting a limit on the amount of words and that is frequency distribution of words
word_features = list(all_words.keys())[:3000]

#finding features that we will be using
def find_features(document):
    words = set(document)    #every single word of the document not their count 
    features = {}
    for w in word_features:     #w in 3000most popular words
        features[w] = (w in words)   #return true in features[w] if w exists in word_features list of top 3000 words 

    return features

#print the result
#print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]   #return a dict of the top 3000 words with the true or false depending on weather they exists in the document or not




#NAIVE BAYES
"""
Basic algorithm which can be scaled and is not complex in application
posterior = prior occurences x likelihood / evidence
Really easy to understand
Uses nltk
It is not that accurate and reliable and gives different results on different runs
"""
training_set = featuresets[:1900]
test_set = featuresets[1900:]

#step 2 === readinf froma saved pickle file
classifier_f = open ("naivebayes.pickle","rb") #---> important to mention rb or wb in python3
classifier = pickle.load(classifier_f)
classifier_f.close()

#classifier = nltk.NaiveBayesClassifier.train(training_set)
##print ("Orignal Naive Bayes Also accuracy: ", (nltk.classify.accuracy(classifier, test_set))*100)
##classifier.show_most_informative_features(15)   #----->shows top 15 most popuar words both sides  
##
##
###scikit incorporation
##MNB_classifier = SklearnClassifier(MultinomialNB())
##MNB_classifier.train(training_set)
##print ("MNB_classifier Also accuracy: ", (nltk.classify.accuracy(MNB_classifier, test_set))*100) 
##
###GaussianNB, BernoulliNB
####GaussianNB_classifier = SklearnClassifier(GaussianNB())
####GaussianNB_classifier.train(training_set)
####print ("GaussianNB_classifier Also accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, test_set))*100) 
##
##BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
##BernoulliNB_classifier.train(training_set)
##print ("BernoulliNB_classifier Also accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)

# LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("LogisticRegression_classifier Also accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100) 
##LogisticRegression_classifier.show_most_informative(15)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print ("SGDClassifier_classifier Also accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100) 

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print ("SVC_classifier Also accuracy: ", (nltk.classify.accuracy(SVC_classifier, test_set))*100) 

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("LinearSVC_classifier Also accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100) 

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print ("NuSVC_classifier Also accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100) 



