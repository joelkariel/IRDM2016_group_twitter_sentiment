import glob
import os
import json
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import time 
import osgeo.ogr
import shapely.geometry
import shapely.wkt
import warnings

from Tweet import Tweet
from Tweet_classified import Tweet_classified
from Crime import Crime
from plot_london import plot_borough_sentiment
from datetime import datetime as dt
from datetime import timedelta
from pyproj import Proj, transform


warnings.filterwarnings('ignore')
def main(classified_tweet_path, crime_path):
    # Load data
    tweets = load_classif_tweets(classified_tweet_path)
    crimes = load_crime_data(crime_path)
    # 1) Split by grid 
    q_count, q_pos_count, q_coord =  grid_split_locations(tweets)
    crime_count = grid_split_crime(crimes,q_coord)
    output_grid_results(q_count, q_pos_count,crime_count)
    
    # 2) Split by borough
    ldn_borough_tweets = split_by_borough(tweets)
    output_borough_results(ldn_borough_tweets)
    
    # 3) Visualise borough sentiment
    plot_borough_sentiment()

def load_crime_data(path):
    ''' Load in crime data .csv from path'''
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
    '''Load in classified tweets from path returning list of tweet_classified objects'''
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

# ----------------------------------   Part 1: Splitting by grid
def grid_split_locations(tweets):
    ''' Split the tweet data set by grid partition based on geolocations attached'''
    min_lat  = float('inf')
    max_lat  =  float('-inf')
    min_long =  float('inf')
    max_long =  float('-inf')
    bounds = [52,49.5,1,-1] #Confine to london area
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
    '''Split the crime data by grid partition and count the number of crimes'''
    crime_count = np.zeros((len(q_coord),1))
    for crime in crimes:
        quad  = quadrant_allocate(q_coord,crime.longitude,crime.latitude)
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

def output_grid_results(quadrant_count, quadrant_pos_count,crime_count):
    '''Computes proportional sentiment for a quadrant and outputs it as a JSON'''
    for q in range(len(quadrant_count)):
        proportional_sentiment = float(quadrant_pos_count[q]/quadrant_count[q])
        print '[Quadrant 1] Total Tweets:',quadrant_count[q][0],'% Positive','{0:.00f}%'.format(proportional_sentiment * 100.),'#Crimes:', crime_count[q][0]

        
# --------------------------------------PART 2: SPLITTING BY BOROUGH
def split_by_borough(tweets):
    MAX_DISTANCE = 0.01 #using buffer technique:area to buffer around borough polygon
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

def output_borough_results(ldn_borough):
    '''Outputs tweet count and positive tweet count for each borough as JSON'''
    with open('./results/borough_tweet_results.json', 'w') as f:
        json.dump(ldn_borough, f)
    for name,dict_count in ldn_borough.items():
        print name, dict_count 
        
if __name__ == "__main__":
    # Place the folder containing your twitter data as the first parameter (leave the r in)
    # Place the folder containing your police data as the second parameter (leave the r in)
    path = os.getcwd()
    twitter_path = path + '\\twitter_classified'
    crime_path = path + '\\crime'
    main(twitter_path,crime_path)
    
    
    
  