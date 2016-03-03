class Crime:
    def __init__(self, crime_id, month, reported_by, falls_within, longitude, latitude, location, lsoa_code, lsoa_name,
                 crime_type, last_outcome, context):
        self.crime_id = crime_id
        self.month = month
        self.reported_by = reported_by
        self.falls_within = falls_within
        self.longitude = longitude
        self.latitude = latitude
        self.location = location
        self.lsoa_code = lsoa_code
        self.lsoa_name = lsoa_name
        self.crime_type = crime_type
        self.last_outcome = last_outcome
        self.context = context
