'''
# installing unidecode
# install nltk.download()
'''

# Packages
import random
import pickle
import nltk
import json
import os
# ----------------
import glob
from Tweet import Tweet
from Main import extract_hashtags, load_crime_data, load_tweets
# ----------------
# String Encoding
from unidecode import unidecode
# ----------------
# NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI #so we can inherit from the nltk classifier class
from nltk.tokenize import word_tokenize
# ----------------
# Machine Learning
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
# ----------------
# Statistics
from statistics import mode #how we are going to choose who got the most votes
# _________________________________________________________
from VoteClassifier import VoteClassifier


print 'Loading Data...'	   
short_pos = open("Training/positive.txt","r").read()
short_neg = open("Training/negative.txt","r").read()
print 'Encoding data...'
# Encode the file to something that can be parsed
short_pos = unicode(short_pos, errors='ignore')
short_neg = unicode(short_neg, errors='ignore')

print '---------------------'
print 'Training in progress...'
print '---------------------'

documents = []
all_words = []

#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J","R","V"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("Pickled/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:10000] 

save_word_features = open("Pickled/word_features10k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("Pickled/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_word_features.close()

random.shuffle(featuresets)
print(len(featuresets))
   
training_set = featuresets[30000:]
testing_set =  featuresets[:20000]

# --------------------------- SAVE TRAINED ALGOS ---------------------------
# --------------------------- CAN USE THEM LATER ---------------------------

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(25)

##
save_classifier = open("Pickled/originalnaivebayes50k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
##


# ----------------------------------------------------

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

##
save_classifier = open("Pickled/MNB_classifier50k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()
##



# ----------------------------------------------------

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

##
save_classifier = open("Pickled/BernoulliNB_classifier50k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()
##



# ----------------------------------------------------

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

##
save_classifier = open("Pickled/LogisticRegression_classifier50k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()
##



# ----------------------------------------------------

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##
save_classifier = open("Pickled/SGDC_classifier50k.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()
##



# ----------------------------------------------------

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

##
save_classifier = open("Pickled/LinearSVC_classifier50k.pickle","wb")
pickle.dump(LinearSVC_classifier , save_classifier)
save_classifier.close()
##



# ----------------------------------------------------

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

##
save_classifier = open("Pickled/NuSVC_classifier50k.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()
##

# ----------------------------------------------------

print 'Training completed...'
print 'Pickling completed... '