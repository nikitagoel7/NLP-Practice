import nltk
import random   #shuffle the dataset from highly ordered to unordered
from nltk.corpus import movie_reviews
import pickle

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

#go to words_features.py
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
training_set = featuresets[:1900]
test_set = featuresets[1900:]

#new
#step 2 === reading from a saved pickle file
classifier_f = open ("naivebayes.pickle","rb") #---> important to mention rb or wb in python3
classifier = pickle.load(classifier_f)
classifier_f.close()

#classifier = nltk.NaiveBayesClassifier.train(training_set)
print ("Naive Bayes Also accuracy: ", (nltk.classify.accuracy(classifier, test_set))*100)

classifier.show_most_informative_features(15)   #----->shows top 15 most popuar words both sides  

#step 1 === writing in the pickle file
###just saving the classfier in a pickle file that is saving our analysis of data
##save_classifier = open ("naivebayes.pickle","wb")
##pickle.dump(classifier, save_classifier)
##save_classifier.close()
##


