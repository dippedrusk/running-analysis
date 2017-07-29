# CMPT 318 Final Project
# Author: Vasundhara Gautam

# butterworth_distance.py reads in and parses a single XML file
# at a time and returns the distance between contiguous GPS
# coordinates smoothed using the Butterworth filter

import numpy as np
import pandas as pd
import math
# Imports for 

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def get_data(inputdata):
   
    from xml.dom.minidom import getDOMImplementation, parse

    gpsdom = parse(inputdata)  # parse an XML file by name
    
    gpscoords = gpsdom.getElementsByTagName("trkpt")
    
    lat = []
    lon = []
    
    for entry in gpscoords:
        lat.append(float(entry.getAttribute("lat")))
        lon.append(float(entry.getAttribute("lon")))
        
    parsed_data = pd.DataFrame([lat,lon], ["lat", "lon"]).transpose()
    return parsed_data

def distance(data): # returns distance in m
    
    nextdata = pd.DataFrame([data['lat'],data['lon']], ["nextlat", "nextlon"]).transpose().shift(periods=-1, axis=0)
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
    # Return Butterworth-smoothed data
    return
   
def main(inputfile):
    points = get_data(inputfile)
    smoothed_points = smooth(points)
    return distance(smoothed_points)