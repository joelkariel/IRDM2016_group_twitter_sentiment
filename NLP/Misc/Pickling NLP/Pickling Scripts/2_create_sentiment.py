
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


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


documents_f = open("Pickled/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("Pickled/word_features10k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

featuresets_f = open("Pickled/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


## LOAD THE ALGOS

open_file = open("Pickled/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/SGDC_classifier5k.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

open_file = open("Pickled/NuSVC_classifier5k.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()


## END LOADING


voted_classifier = VoteClassifier( classifier,
                                   NuSVC_classifier, 
                                   LinearSVC_classifier,
								   SGDClassifier_classifier,
								   MNB_classifier,
								   BernoulliNB_classifier,
								   LogisticRegression_classifier )

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),float(voted_classifier.confidence(feats))


## Here we will go through all the objects and append them to the data_collection 
tweet_path = r'C:\Users\Andrew\Downloads\Twitter'
#tweet_path = r'C:\Users\Andrew\Documents\IRDM\group_backup\NLP\Tweets_Subset'

# Returns a bunch of objects
tweets_objects = load_tweets(tweet_path)

file_path = 'C:\Users\Andrew\Documents\IRDM\GitHub\irdm_twitter_sentiment\NLP\Optimised'
for instance in range(0,len(tweets_objects)):
	try:
		classify = sentiment(tweets_objects[instance].text.encode('utf-8'))
		category = classify[0]
		confidence = classify[1]
		if not os.path.exists( file_path + '\\' +'classified_tweets.txt'):
				open( file_path + '\\' + 'classified_tweets.txt', 'w').close # Creates the log file
		with open( file_path + '\\' + 'classified_tweets.txt', 'a' ) as outfile:
			outfile.write( str(category) + ',' ) #Sentiment Classification
			outfile.write( str(confidence) + ',' ) #How sure we are of this classification
			outfile.write( str(tweets_objects[instance].tweet_id) + ',' ) #Tweet ID
			outfile.write( str(tweets_objects[instance].text.encode('utf-8')) + ',' ) #Tweet
			outfile.write( str(tweets_objects[instance].username.encode('utf-8')) + ',' ) #Username
			#outfile.write( str(tweets_objects[instance].date) + ',' ) #Date
			outfile.write( str(tweets_objects[instance].timestamp) + ',' ) #Datetime 
			outfile.write( str(tweets_objects[instance].raw_unix) + ',' ) #raw_unix
			outfile.write( str(tweets_objects[instance].latitude) + ',' ) #latitude
			outfile.write( str(tweets_objects[instance].longitude) + '\n' ) #longitude
		print 'classified tweet number ' + str(instance)
	except:
		## THE ERRORS CAN BE WRITTEN TO A FILE
		print 'ERROR ENCOUNTERED'
		if not os.path.exists( file_path + '\\' + 'error_log.txt'):
				open( file_path + '\\' + 'error_log.txt', 'w').close # Creates the error log file
		with open( file_path + '\\' + 'error_log.txt', 'a' ) as outfile:
			outfile.write( 'Tweet ID: ' + str(tweets_objects[instance].tweet_id) + '\n' )
			outfile.write( 'Tweet: ' + str(tweets_objects[instance].text.encode('utf-8')) + '\n' )
			outfile.write( 'Username: ' + str(tweets_objects[instance].username.encode('utf-8')) + '\n' )
			outfile.write( 'Timestamp (Unix): ' + str(tweets_objects[instance].raw_unix) + '\n' )
			outfile.write( 'Date: ' + str(tweets_objects[instance].timestamp) + '\n' )
			outfile.write( 'latitude: ' + str(tweets_objects[instance].latitude) + '\n' )
			outfile.write( 'longitude: ' + str(tweets_objects[instance].longitude) + '\n' )
			outfile.write( '-------------------------------------------' + '\n' )