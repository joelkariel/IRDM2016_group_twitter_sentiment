'''

NEXT WEEK I WILL PICKLE THE CLASSIFIERS ON THE LARGE DATASETS
++ THIS ALLOWS US TO SAVE THE CLASSIFIERS AND THEN USE THEM OVER AND OVER

# installing unidecode
# install nltk.download

'''

# Packages
import random
import pickle
import nltk
import json
import os

# String Encoding
from unidecode import unidecode

# NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI #so we can inherit from the nltk classifier class
from nltk.tokenize import word_tokenize

# Machine Learning
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Statistics
from statistics import mode #how we are going to choose who got the most votes


# TRAINING

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
		self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes)) #ENSURE THIS IS OUT OF THE FOR LOOP OTHERWISE I GET SHIT CONF VALS
		conf = float(choice_votes) / float(len(votes))
		return conf
        
short_pos = open("short_positive.txt","r").read()
short_neg = open("short_negative.txt","r").read()

# Encode the file to something that can be parsed
short_pos = unicode(short_pos, errors='ignore')
short_neg = unicode(short_neg, errors='ignore')

print '---------------------'
print type(short_pos)
print type(short_neg)
print '---------------------'

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000] #we have 5000 features

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets) #shuffle the sets so its not just pos / neg / pos / neg etc

# positive data example:      
##training_set = featuresets[:10000]
##testing_set =  featuresets[10000:]

##
### data ---- note as this is shuffled there is no big deal      
training_set = featuresets[250:]
testing_set =  featuresets[:250]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier( classifier,NuSVC_classifier,LinearSVC_classifier,SGDClassifier_classifier,
                                   MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier )
								  
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)



# TRAINING COMPLETE - NOW TO COMPUTE SENTIMENT

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),float(voted_classifier.confidence(feats))
	
#print(sentiment("This movie was shit"))

# ===========================================================================================
# PICK UP OUR TEST SET FOR CLASSIFICATION
data_collection = []
with open('sample.json') as f:
	s = f.read()
	data = json.loads(s)
	for tweet in data['tweets']:
		text = tweet['text'].encode('utf-8') #Added to encode correctly
		tweetId = tweet['tweetId']
		username = tweet['username']
		date = tweet['date']
		latitude = tweet['latitude']
		longitude = tweet['longitude']
		data_collection.append( [ tweetId, text, username, date, latitude, longitude ] )

'''
++ NOTE: I HAVE HAD TO PUT A TRY AND EXCEPT IN THIS CODE AS 
		 THE TEXT FILE IS NOT CLEAN AND PARSED FROM 
		 ISSUES WITH UNICODE ETC ETC ...
'''
file_path = 'C:\Users\Andrew\Documents\IRDM\group_backup\NLP'
for instance in range(0,len(data_collection)):
	try:
		classify = sentiment(data_collection[instance][1])
		category = classify[0]
		confidence = classify[1]
		if not os.path.exists( file_path + '\\' +'classified_tweets.txt'):
				open( file_path + '\\' + 'classified_tweets.txt', 'w').close # Creates the log file
		with open( file_path + '\\' + 'classified_tweets.txt', 'a' ) as outfile:
			outfile.write( str(category) + ',' ) #Sentiment Classification
			outfile.write( str(confidence) + ',' ) #How sure we are of this classification
			outfile.write( str(data_collection[instance][0]) + ',' ) #Tweet ID
			outfile.write( str(data_collection[instance][1]) + ',' ) #Tweet
			outfile.write( str(data_collection[instance][2]) + ',' ) #Username
			outfile.write( str(data_collection[instance][3]) + ',' ) #Date
			outfile.write( str(data_collection[instance][4]) + ',' ) #latitude
			outfile.write( str(data_collection[instance][5]) + '\n' ) #longitude
		print 'classified tweet number ' + str(instance)
	except:
		## THE ERRORS CAN BE WRITTEN TO A FILE
		print 'ERROR ENCOUNTERED'
		if not os.path.exists( file_path + '\\' + 'error_log.txt'):
				open( file_path + '\\' + 'error_log.txt', 'w').close # Creates the error log file
		with open( file_path + '\\' + 'error_log.txt', 'a' ) as outfile:
			outfile.write( 'Tweet ID: ' + str(data_collection[instance][0]) + '\n' )
			outfile.write( 'Tweet: ' + str(data_collection[instance][1]) + '\n' )
			outfile.write( 'Username: ' + str(data_collection[instance][2]) + '\n' )
			outfile.write( 'Date: ' + str(data_collection[instance][3]) + '\n' )
			outfile.write( 'latitude: ' + str(data_collection[instance][4]) + '\n' )
			outfile.write( 'longitude: ' + str(data_collection[instance][5]) + '\n' )
			outfile.write( '-------------------------------------------' + '\n' )
		