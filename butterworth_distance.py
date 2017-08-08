"""
CMPT 318 Final Project
Author: Vasundhara Gautam

butterworth_distance.py has functions to parse a GPX file at
a time, smooth the coordinates using the Butterworth filter,
and return the distance run in a given file.
"""

import numpy as np
import pandas as pd
from scipy import signal
from datetime import date, datetime
from xml.dom.minidom import getDOMImplementation, parse


"""
Parses XML file with GPS data and returns a DataFrame with latitudes,
longitudes, and timestamps.
"""
def get_data(inputfile):
    gpsdom = parse(inputfile)
    gpscoords = gpsdom.getElementsByTagName("trkpt")
    time = gpsdom.getElementsByTagName("time")
    length = gpsdom.getElementsByTagName("time").length
    lat = []
    lon = []
    timestamps = []
    for i in range(0,length-1):
        lat.append(float(gpscoords[i].getAttribute("lat")))
        lon.append(float(gpscoords[i].getAttribute("lon")))
        timestamps.append(str(time[i].firstChild.nodeValue))
    parsed_data = pd.DataFrame([timestamps,lat,lon], ["time", "lat", "lon"]).transpose()
    parsed_data['time'] = pd.to_datetime(parsed_data['time'], format='%Y-%m-%dT%H:%M:%SZ')
    return parsed_data


"""
Returns running distance in m between points, given columns of latitude,
longitude, and corresponding timestamps. Excludes any pauses and any walking
or slow running
"""
def distance(data):
    nextdata = pd.DataFrame([data['lat'].values, data['lon'].values, data['time'].values], ["nextlat", "nextlon","nexttime"])
    nextdata = nextdata.transpose().shift(periods=-1, axis=0)
    data = pd.concat([data, nextdata], axis=1)
    data['time'] = pd.to_datetime(data['time'])
    data['nexttime'] = pd.to_datetime(data['nexttime'])
    data['timediff'] = data['nexttime'] - data['time']
    data = data.drop(['nexttime', 'time'], axis=1)
    
    # Formula adapted from https://en.wikipedia.org/wiki/Haversine_formula
    radius = 6371000 # in m
    lat = radify(data['lat']).astype(np.float64)
    lon = radify(data['lon']).astype(np.float64)
    nextlat = radify(data['nextlat']).astype(np.float64)
    nextlon = radify(data['nextlon']).astype(np.float64)
    
    a = 2*radius
    b = (np.sin((nextlat-lat)/2))**2
    c = np.cos(lat) * np.cos(nextlat) * ((nextlon-lon)/2)**2

    data['distbetween'] = a*np.arcsin(np.sqrt(b+c))
    
    # Want to exclude any pauses (gaps in GPS data longer than 8 seconds)
    # and any slow running (pace greater than 0.67 s/m, same as speed
    # less than 1.5 m/s)
    data['pace'] = data['timediff'] / data['distbetween']
    
    maximumtime = pd.Timedelta(seconds=8)
    maximumpace = pd.Timedelta(seconds=0.67)

    data = data[(data['timediff'] <= maximumtime) | (data['pace'] <= maximumpace)]
    
    totaldistance = pd.DataFrame.sum(data['distbetween'], axis=0)
    return totaldistance


"""
Converts degrees of latitude and longitude to radians
"""
def radify(column):
    return column * np.pi / 180


"""
Returns Butterworth-smoothed data
"""
def smooth(data):
    b, a = signal.butter(3, 0.3, btype='low', analog=False)
    lat = signal.filtfilt(b, a, data['lat'])
    lon = signal.filtfilt(b, a, data['lon'])
    low_passed = pd.DataFrame([lat, lon], ["lat", "lon"]).transpose()
    return low_passed

    
"""
Takes a .gpx filename as input and returns the distance after smoothing
"""
def get_distance(inputfile):
    points = get_data(inputfile)
    latlon = pd.DataFrame([points['lat'], points['lon']]).transpose()
    latlon_smoothed = smooth(latlon)
    points.drop(labels=['lat', 'lon'], axis=1, inplace=True)
    points = pd.concat([points, latlon_smoothed], axis=1)
    return distance(points)

"""
Takes a .gpx filename as input and returns the difference in minutes
between the first and last timestamps to give the duration of exercise.
This function does not return the duration that I was actually running
above a certain speed, as explained in the report.
"""
def get_time(inputfile):
    gpsdom = parse(inputfile)
    gpscoords = gpsdom.getElementsByTagName("trkpt")
    time = gpsdom.getElementsByTagName("time")
    length = gpsdom.getElementsByTagName("time").length
    starttime = datetime.strptime(time[0].firstChild.nodeValue, '%Y-%m-%dT%H:%M:%SZ')
    endtime = datetime.strptime(time[length-1].firstChild.nodeValue, '%Y-%m-%dT%H:%M:%SZ')
    timediff = endtime-starttime
    return float(timediff.seconds/60.0) # timediff does not have a .minutes attribute