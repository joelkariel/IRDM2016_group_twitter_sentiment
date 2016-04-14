import datetime

class Tweet_classified:
    def __init__(self, classification,confidence, date, latitude, longitude,OS1936lat,OS1936long):
        self.timestamp = datetime.datetime.fromtimestamp(int(date / 1000))
        self.latitude = latitude
        self.longitude = longitude
        self.OS1936lat = OS1936lat
        self.OS1936long = OS1936long
        self.classif = classification
        self.conf = confidence
        
    def assign_sentiment(self,classification,confidence):
        '''Add and store sentiment of classified tweets '''
        self.classif = classification
        self.conf = confidence
    
