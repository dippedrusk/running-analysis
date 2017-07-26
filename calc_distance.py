import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
import math
import sys

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
   
def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))

    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
