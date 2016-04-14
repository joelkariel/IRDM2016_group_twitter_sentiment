from collections import OrderedDict
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fiona
import json
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon, shape

def plot_borough_sentiment():
    ''' Function which takes JSON output of borough tweet sentiment and .shp file of borough
        and plots a colour map of sentiment'''
    # 1) Open London borough boundaries from .shp file
    path = os.getcwd()
    shape_file_name = path + '\\LDN_boundary_data\London_Borough_Excluding_MHW.shp'
    positive_tweets = {}
    pos_tweets = []
    # 2) Open results JSON file from classified tweets and extract sentiment for each borough
    with open(path+ "\\results\\borough_tweet_results.json") as json_file:
        data = json.load(json_file, object_pairs_hook=OrderedDict)
        for key, value in data.items():
            positive_tweets[str(key)] = float(value['pos_count'])/value['count']
    # 3) Extract London borough boundaries from .shp file
    pos_tweets = []
    for pol in fiona.open(shape_file_name):
        borough_name = pol['properties']['NAME']
        value = data[str(borough_name)]
        pos_tweets.append(float(value['pos_count'])/value['count'])
    mp = MultiPolygon(
        [shape(pol['geometry']) for pol in fiona.open(shape_file_name)])

    # 4) Iterate through boroughs (patches) and assign colour scheme based on +ve sentiment
    patches = []
    cmap = plt.get_cmap('Blues')
    max_s = max(pos_tweets) #max sentiment across all boroughs 
    min_s = min(pos_tweets) #min sentiment across all boroughs 
    for idx, p in enumerate(mp):   
        new_percentage = float(pos_tweets[idx]-min_s)/(max_s-float(min_s))
        colour = cmap(new_percentage)
        #colour = cmap(pos_tweets[idx]) 
        patches.append(PolygonPatch(p,fc=colour, ec='#111111',lw=.8, alpha=1., zorder=4 ))

    # 5) Set plot configurations
    figwidth = 14
    h = 200933.9 - 155850.8
    w = 561957.5 - 503568.2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    minx, miny, maxx, maxy = mp.bounds
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.1*w, maxx + 0.1*w)
    ax.set_ylim(miny - 0.1 * h, maxy + 0.1* h)
    ax.set_aspect(1)

    
    breaks = np.linspace(min_s,max_s,5)
    labels = ['> 0 +ve Sentiment']+["> %.1f %% +ve \nsentiment"%(perc*100) for perc in breaks[:-1]]
    def custom_colourbar(cmap, ncolors, labels, **kwargs):    
        """Create a custom, discretized colorbar with correctly formatted/aligned labels.
        
        cmap: the matplotlib colormap object you plan on using for your graph
        ncolors: (int) the number of discrete colors available
        labels: the list of labels for the colorbar. Should be the same length as ncolors.
        """
        from matplotlib.colors import BoundaryNorm
        from matplotlib.cm import ScalarMappable
            
        norm = BoundaryNorm(range(0, ncolors), cmap.N)
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors+0.5)
        colorbar = plt.colorbar(mappable, **kwargs)
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors+1)+0.5)
        colorbar.set_ticklabels(range(0, ncolors))
        colorbar.set_ticklabels(labels)
        return colorbar

    colour_bar = custom_colourbar(cmap, ncolors=len(labels)+1, labels=labels)
    colour_bar.ax.tick_params(labelsize =12)

    # 6) Plot boroughs 
    pc = PatchCollection(patches, match_original=True)  
    ax.add_collection(pc)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Aggregate Twitter Sentiment by London Borough", fontdict={'size':15}, y=1.05)
    plt.savefig('results/london_from_shp.png', alpha=True, dpi=300)
    plt.show()
    
   





   

