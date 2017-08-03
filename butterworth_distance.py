# CMPT 318 Final Project
# Author: Vasundhara Gautam

# butterworth_distance.py reads in and parses a single XML file
# at a time and returns the distance between contiguous GPS
# coordinates smoothed using the Butterworth filter

import numpy as np
import pandas as pd
import math
# Imports for butterworth

from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

def get_data(inputdata):
    from xml.dom.minidom import getDOMImplementation, parse
    gpsdom = parse(inputdata)  # parse an XML file by name
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

def distance(data): # returns distance in m
    nextdata = pd.DataFrame([data['lat'], data['lon'], data['time']], ["nextlat", "nextlon","nexttime"])
    nextdata = nextdata.transpose().shift(periods=-1, axis=0)
    data = pd.concat([data, nextdata], axis=1)
    data['time'] = pd.to_datetime(data['time'])
    data['nexttime'] = pd.to_datetime(data['nexttime'])
    data['timediff'] = data['nexttime'] - data['time']
    data = data.drop(['nexttime', 'time'], axis=1)
    
    radius = 6371000 # m
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

def radify(column):
    return column * np.pi / 180
 
def distancebetween2points(data): # returns distance in m
    radius = 6371000
    lat = radify(data['lat'])
    lon = radify(data['lon'])
    nextlat = radify(data['nextlat'])
    nextlon = radify(data['nextlon'])
    a = 2*radius
    b = (math.sin((nextlat-lat)/2))**2
    c = math.cos(lat)*math.cos(nextlat)*((nextlon-lon)/2)**2
    dist = a*math.asin(math.sqrt(b+c))
    return dist

def smooth(data):
    # Return Butterworth-smoothed data
    from scipy import signal
    b, a = signal.butter(3, 0.1)#, btype='lowpass', analog=False)
    lat = signal.filtfilt(b, a, data['lat'])
    lon = signal.filtfilt(b, a, data['lon'])
    low_passed = pd.DataFrame([data['time'], lat, lon], ["time", "lat", "lon"])
    low_passed = low_passed.transpose()
    return low_passed
   
def get_distance(inputfile):
    points = get_data(inputfile)
    res = distance(points)
    print('Before filtering: %0.2f' % res)
    points = smooth(points)
    res = distance(points)
    print('After filtering: %0.2f' % res)
    return res