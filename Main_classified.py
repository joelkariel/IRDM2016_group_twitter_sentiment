import glob
import os
import json
import csv
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import time 
from Tweet import Tweet
from Tweet_classified import Tweet_classified
from Crime import Crime
from datetime import datetime as dt
from datetime import timedelta
from pyproj import Proj, transform
import osgeo.ogr
import shapely.geometry
import shapely.wkt
import warnings




warnings.filterwarnings('ignore')
def main(classified_tweet_path, crime_path):
    
    tweets = load_classif_tweets(classified_tweet_path)
    crimes = load_crime_data(crime_path)
    ldn_borough = split_regions(tweets)
    #extract_hashtags(tweets)
    #tweet_timeseries = build_tweet_timeseries(tweets)
    #crime_timeseries = build_crime_timeseries(crimes)
    #plot_timeseries(tweet_timeseries,'Plot of Tweet timeseries')
    #q_count, q_pos_count, q_coord =  grid_split_locations(tweets)
    #crime_count = grid_split_crime(crimes,q_coord)
    #output_results(q_count, q_pos_count,crime_count)
    output_borough_results(ldn_borough)
def split_regions(tweets):
    MAX_DISTANCE = 0.01 #using buffer technique: approx
    ldn_borough= {}
    ldn_borough_area = {}
    # Import .shp file with boundaries
    path = os.getcwd()
    shapefile = osgeo.ogr.Open(path + '\\LDN_boundary_data\London_Borough_Excluding_MHW.shp')
    layer = shapefile.GetLayer(0)
    geo_ref  = layer.GetSpatialRef()
    
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        name = feature.GetField("NAME")
        geometry = feature.GetGeometryRef()
        shape = shapely.wkt.loads(geometry.ExportToWkt()) 
        ldn_borough_area[name] = shape.buffer(MAX_DISTANCE)
        ldn_borough[name] = {'count':0,'pos_count':0}
    count = 0
    for tweet in tweets:
        pt = shapely.geometry.Point(tweet.OS1936lat,tweet.OS1936long)
        count +=1
        for borough_name,borough_area in ldn_borough_area.items():
            # Print progress
            
            if count % 100000 == 0:
                print count, " Tweets Classified"
            #Allocate tweet
            if pt.within(borough_area):
                #print  "Tweet is in or near " + borough_name
                ldn_borough[borough_name]['count'] += 1
                if tweet.classif == 'pos':
                    ldn_borough[borough_name]['pos_count'] +=1 
                break
    return ldn_borough
            
             
        
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
                    if entry[4] and entry[5]:
                        lat  = float(entry[4])
                        long = float(entry[5])
                        crime = Crime(entry[0], entry[1], entry[2], entry[3],
                                  lat, long, entry[6], entry[7],
                                  entry[8], entry[9], entry[10], entry[11])
                        crimes.append(crime)
    return crimes
    
def load_classif_tweets(path):
    tweets = []
    num_files = len(glob.glob1(path, "*.txt"));
    print num_files
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        if counter % 250 == 0:
            print "Parsing file: " + str(counter) + "/" + str(num_files)

        counter += 1
        with open(os.path.join(path, filename)) as file:
            file = file.read().splitlines()
            tweet_reader = csv.reader(file)
            counter2 = 0
            for row in tweet_reader:
                try:
                    classification = row[0]
                    confidence = row[1]
                    
                    lat = float(row[7])
                    long = float(row[8])
                    date = int(row[6])
                    #conversion for comparison with LDN ordanance survey geo projection coordinates
                    inProj = Proj(init='epsg:4326')
                    outProj = Proj(init='epsg:7405')
                    try:
                        # build tweet object
                        OS1936lat,OS1936long = transform(inProj,outProj,long,lat)
                        tweet = Tweet_classified(classification,confidence, date, lat, long,OS1936lat,OS1936long)
                        tweets.append(tweet)
                        #print OS1936lat,OS1936long
                        #sys.stdout.flush()
                    except RuntimeError:
                        #tweet outside of UK
                        OS1936lat = None
                        OS1936long = None 
                    counter2 += 1
                    if counter2%10000 == 0:
                        print counter2, '(classified)tweets read'
                        sys.stdout.flush()
                except IndexError:
                    #print 'Unreadable line'
                    pass
    return tweets
                
                
        


def plot_timeseries(ts,title):
	dates  = ts[:,0]
	values = ts[:,1]
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel('Day')
	ax.set_ylabel('Tweets')
	ax.plot(dates,values)

	plt.gcf().autofmt_xdate()
	#show graph
	plt.show()
	
	

def build_tweet_timeseries(tweets):
	
	#Get day range of dataset
	current_max = tweets[0].timestamp
	current_min = tweets[0].timestamp
	for tweet in tweets:
		if tweet.timestamp > current_max:
			current_max = tweet.timestamp
		if tweet.timestamp < current_min:
			current_min = tweet.timestamp
	first_date = current_min.date()
	last_date = current_max.date()
	print 'First: ', first_date, ' Last: ', last_date
	time_window = (last_date- first_date).days
	tweet_count_series = []
	# Count number of tweets on each day
	for d in range(0,time_window+1):
		curr_day = first_date + timedelta(days=d)
		tweet_count = 0
		# Count number of tweets if date matches current day
		tweet_count = sum(1 if tweet.timestamp.date() == curr_day else 0 for tweet in tweets)
		tweet_count_series.append([curr_day,tweet_count])
		print 'Day:',d,'of',time_window
		sys.stdout.flush()
	tweet_ts = np.array(tweet_count_series)
	
	#---------------------Write Output for Excel Plots-----------
	np.savetxt('tweet_timeseries.csv',tweet_ts,delimiter=",",fmt=['%s','%1.3f'])
	return tweet_ts

def build_crime_timeseries(crimes):
	print crimes[0].month


def grid_split_locations(tweets):

    ''' Split the tweet data set by grid partition based on geolocations attached'''
    min_lat  = float('inf')
    max_lat  =  float('-inf')
    min_long =  float('inf')
    max_long =  float('-inf')
    bounds = [52,49.5,1,-1]
    for tweet in tweets:
        if tweet.latitude > bounds[0] or tweet.latitude < bounds[1]:
            break
        elif tweet.longitude > bounds[2] or tweet.longitude < bounds[3]:
            break
        else:
            if tweet.latitude  < min_lat: 
                min_lat  = tweet.latitude
            if tweet.latitude  > max_lat:
                max_lat  = tweet.latitude
            if tweet.longitude < min_long:
                min_long = tweet.longitude
            if tweet.longitude > max_long:
                max_long = tweet.longitude 

    print 'Map ranges: ',float(min_lat),float(min_long),float(max_lat),float(max_long)
    # 2) Choose number of splits of grid and find quadrant coordinates
    n_split = 7
    lat_step  = (max_lat  - min_lat) /n_split 
    long_step = (max_long - min_long)/n_split

    quadrant_list = np.zeros((n_split*n_split,4))
    counter = 0
    for x in range(n_split):
        for y in range(n_split):
            # Format [lower-lat, upper-lat, lower-long,upper-long]
            quadrant = [(min_lat + x*lat_step), (min_lat + (x+1)*lat_step),
                        (min_long+ y*long_step),(min_long+ (y+1)*long_step)]
            
            quadrant_list[counter,:] = quadrant
            counter += 1
      
    print quadrant_list

    # 3) Allocate tweet to quadrant
    quadrant_count = np.zeros((n_split*n_split,1))
    quadrant_pos_count = np.zeros((n_split*n_split,1))

    for tweet in tweets:
        quad = quadrant_allocate(quadrant_list,tweet.longitude,tweet.latitude)
        if quad != None:
            if tweet.classif == 'pos':
                quadrant_pos_count[quad] += 1
            quadrant_count[quad] +=1
        
    
    return quadrant_count, quadrant_pos_count, quadrant_list

def grid_split_crime(crimes,q_coord):
    crime_count = np.zeros((len(q_coord),1))
    for crime in crimes:
        quad  = quadrant_allocate(q_coord,crime.longitude,crime.latitude)
        #print crime.longitude, crime.latitude, quad
        crime_count[quad] += 1
    return crime_count
    
    
def quadrant_allocate(quadrant_list,longitude, latitude):
    '''For a given long/lat pair determines which quadrant it belongs to'''
    #quadrant list in format [a (lat),b (lat),c (long),d (long)]
    curr_quad = 0
    for quad in quadrant_list:
        #if within latitude range
        if latitude >= quad[0] and  latitude <= quad[1]:
            if longitude >=quad[2] and  longitude <= quad[3]:
                return curr_quad
        curr_quad += 1

def output_results(quadrant_count, quadrant_pos_count,crime_count):
    # 4) Compute proportional sentiment for region
    for q in range(len(quadrant_count)):
        proportional_sentiment = float(quadrant_pos_count[q]/quadrant_count[q])
        print '[Quadrant 1] Total Tweets:',quadrant_count[q][0],'% Positive','{0:.00f}%'.format(proportional_sentiment * 100.),'#Crimes:', crime_count[q][0]
def output_borough_results(ldn_borough):
    with open('./results/borough_tweet_results.json', 'w') as f:
        json.dump(ldn_borough, f)
    for name,dict_count in ldn_borough.items():
        print name, dict_count 
        
if __name__ == "__main__":
    # Place the folder containing your twitter data as the first parameter (leave the r in)
    # Place the folder containing your police data as the second parameter (leave the r in)
    path = os.getcwd()
    twitter_path = path + '\\twitter_sample'
    crime_path = path + '\\Crime_sample'
    main(twitter_path,crime_path)
    #main(r"C:\Users\Ross\Downloads\Twitter", r"C:\Users\Ross\Dropbox\IRDMGROUP\Crime")
