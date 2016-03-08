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


def main():
	tweet_path = r'C:\Users\Andrew\Downloads\Twitter'
	tweets = load_tweets(tweet_path)
	print tweets 
	
if __name__ == "__main__":
	main()