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
    
    
    
    data['distbetween'] = data.apply(distancebetween2points, axis=1)
    totaldistance = pd.DataFrame.sum(data, axis=0)
    return totaldistance['distbetween']

def radify(number):
    return number * math.pi / 180
    
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
    # Return Butterworth-smoothed data, eventually
    dim=2
    observation_stddev = 20/100000
    transition_stddev = 10/100000
    initial_state = data.iloc[0]
    observation_covariance = observation_stddev**2 * np.identity(dim)
    transition_covariance = transition_stddev**2 * np.identity(dim)
    transition_matrix = np.identity(dim)
    
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    
    kalman_smoothed, _ = kf.smooth(data)
    smoothed_data = pd.DataFrame(kalman_smoothed)
    smoothed_data.columns = ['lat','lon']
    return smoothed_data
   
def main(inputfile):
    points = get_data(inputfile)
    smoothed_points = smooth(points)
    return distance(smoothed_points)