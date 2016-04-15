====================================
Parser (see Main.py):
====================================

The parser offers the following functionality (uncomment the methods for your required use):
> tweets = load_tweets(tweet_path)
This loads the tweets from the raw data and places them into a collection of tweet objects. 
> crimes = load_crime_data(crime_path)
This loads the crime from the raw data and places them into a collection of crime objects. 
> store_crimes_as_geojson(crimes, tweets)
This writes out crimes and tweets as geojson (which is then read by the web app). 
> extract_hashtags(tweets)
This identifies the hashtags within the raw tweet data and populates the following properties:
tweet.text_without_hashtags = the tweet with hashtags removed
tweet.hashtags = [A list of hashtags present in the tweet]

====================================
Web app (In 'WebApp' folder):
====================================

The code for the webapp can be found here: https://github.com/joelkariel/irdm_twitter_sentiment/tree/master/WebApp

map.html needs to be run in the browser. 

Note: To run this code you will need to provide your own Google Maps API key.

Reason: Google advises against making unrestricted API keys public (placing them on github). 
As our code is public, we have restricted the API key to only work with our website. 

To obtain a key: https://developers.google.com/maps/documentation/javascript/get-api-key

Working implementation:
A running version of the web app can be found at:
http://s609544430.websitehome.co.uk/IRDM/index.html

Note: As the web app is programmed in JavaScript, heatmap generation is done 
on the client side, and initial generation might take up to 30 seconds.  

====================================
Results Analysis (In '3. Results' folder); see Main_classified.py:
====================================

Usage: ~$ python Main_classified.py

Main_classified.py takes two input arguments (defaults already set) 
> twitter_path = <PATH TO FOLDER CONTAINING CLASSIFIED TWEETS>
> crime_path   = <PATH TO FOLDER CONTAINING CRIME DATA>  

The script performs the following steps on the NLP classified tweet data
1) Loads in the data 
2) Splits the data by a grid system
3) Splits the tweet data by London borough

====================================
Tweet Mining (twitter_text_mining.R):
====================================
The file twitter_text_mining.R allows you to read in Twitter data, analyse it and plot it.

Please note: our Twitter data had already been run through the NLP algorithms to assign each tweet sentiment estimates.

Simply read in your Twitter data as a .txt or .csv.
The sentiment analysis part of the code is at the bottom, and requires your data to have sentiment estimates.

This code creates:
* A term document matrix
* Most frequent term bar chart
* Associations with any term
* Network graph plot
* Dendrogram plot
* Plot of relative sentiment over time
* Grid plot of London (specific to our project)

Most of the code came from this fantastic text mining R tutorial, courtesy of Yanchang Zhao, PhD:
http://www2.rdatamining.com/uploads/5/7/1/3/57136767/rdatamining-slides-text-mining.pdf




