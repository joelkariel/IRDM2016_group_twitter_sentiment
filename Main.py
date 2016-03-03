import glob
import os
import json
import csv

from Tweet import Tweet
from Crime import Crime

def main(tweet_path, crime_path):
    tweets = load_tweets(tweet_path)
    crimes = load_crime_data(crime_path)
    extract_hashtags(tweets)


def extract_hashtags(tweets):
    for tweet in tweets:
        tweet.text_without_hashtags = tweet.text
        if '#' in tweet.text:
            tweet.hashtags = set([i[1:] for i in tweet.text.split() if i.startswith("#")])
            for hashtag in tweet.hashtags:
                tweet.text_without_hashtags = tweet.text_without_hashtags.replace('#' + hashtag, '')


def load_crime_data(path):
    crimes = []
    counter = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            print root, file
            with open(os.path.join(root, file), "r") as data_file:
                reader = csv.reader(data_file, delimiter=',')
                next(reader, None)
                for entry in reader:
                    crime = Crime(entry[0], entry[1], entry[2], entry[3],
                              entry[4], entry[5], entry[6], entry[7],
                              entry[8], entry[9], entry[10], entry[11])
                    crimes.append(crime)
    return crimes


def load_tweets(path):
    tweets = []
    num_files = len(glob.glob1(path, "*.txt"));
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        if counter % 250 == 0:
            print "Parsing file: " + str(counter) + "/" + str(num_files)

        counter += 1
        with open(os.path.join(path, filename)) as data_file:
            data = json.load(data_file)["tweets"]
            for entry in data:
                tweet = Tweet(entry["country"], entry["date"], entry["latitude"], entry["longitude"],
                              entry["placeName"], entry["placeType"], entry["profileLocation"], entry["text"],
                              entry["tweetId"], entry["username"])
                tweets.append(tweet)
    print "Parsing complete"
    return tweets


if __name__ == "__main__":
    # Place the folder containing your twitter data as the first parameter (leave the r in)
    # Place the folder containing your police data as the second parameter (leave the r in)
    main(r"C:\Users\Ross\Downloads\Twitter", r"C:\Users\Ross\Dropbox\IRDMGROUP\Crime")
