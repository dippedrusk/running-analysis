"""
CMPT 318 Final Project
Author: Vasundhara Gautam

running_analysis.py is the main document in this project and running it will
produce all the beautiful graphs explained and analysed in the report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from butterworth_distance import get_distance, get_time
from scipy import stats


"""
Getting the data in
Get the GPS filename list from the CSV file GPSdata.csv, parse the datetime
strings, and create timestamps to be able to do regressions
"""
running_df = pd.read_csv('GPSdata.csv', parse_dates=True, infer_datetime_format=True, encoding='utf-8')
running_df['datetime'] = pd.to_datetime(running_df['datetime']) # Redundancy to be 200% sure this works
# Timestamps are necessary to be able to do regressions
def to_timestamp(inputdatetime):
    return inputdatetime.timestamp()
running_df['timestamp'] = running_df['datetime'].apply(to_timestamp)

"""
Getting smoothed distance for all run files and doing some preliminary
statistics on this data
"""
get_distance = np.vectorize(get_distance)
get_time = np.vectorize(get_time)
running_df['distance'] = get_distance(running_df['filename'])
running_df['duration'] = get_time(running_df['filename'])
running_df['avg_speed'] = running_df['distance'] / (running_df['duration']*60.0)


# Distance: Linear regression and graph with scatter plot and best-fit line
fit = stats.linregress(running_df['timestamp'], running_df['distance'])
print('The p-value of run distance over time is %.3f' % fit.pvalue)

plt.figure(figsize=(12,10))
plt.plot(running_df['datetime'], running_df['distance'], 'r.', markersize=15, label='Runs')
plt.plot(running_df['datetime'], running_df['timestamp']*fit.slope+fit.intercept, 'b-', linewidth=3, label='Best-fit line')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Run distance [m]',fontsize=17)
plt.title('Trend of Increased Running Distance',fontsize=19)
plt.xticks(rotation=40)
plt.legend(loc=4, fontsize=14)
plt.savefig('distance_plot.png')


# Duration: Linear regression and graph with just a scatter plot
fit = stats.linregress(running_df['timestamp'], running_df['duration'])
print('The p-value of run duration over time is %.3f' % fit.pvalue)
plt.figure(figsize=(12,10))
plt.plot(running_df['datetime'], running_df['duration'], 'g.', markersize=15)
plt.plot(running_df['datetime'], running_df['timestamp']*fit.slope+fit.intercept, 'b-', linewidth=3, label='Best-fit line')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Run duration [minutes]',fontsize=17)
plt.title('Scatterplot of Running Duration',fontsize=19)
plt.xticks(rotation=40)
plt.savefig('duration_plot.png')


# Speed: Linear regression and graph with a scatter plot and best-fit line again
fit = stats.linregress(running_df['timestamp'], running_df['avg_speed'])
print('The p-value of average run speed over time is %.3f' % fit.pvalue)
plt.figure(figsize=(12,10))
plt.plot(running_df['datetime'], running_df['avg_speed'], 'm.', markersize=15, label='Runs')
plt.plot(running_df['datetime'], running_df['timestamp']*fit.slope+fit.intercept, 'b-', linewidth=3, label='Best-fit line')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Average run speed [m/s]',fontsize=17)
plt.title('Trend of Increased Running Speed',fontsize=19)
plt.xticks(rotation=40)
plt.legend(loc=4, fontsize=14)
plt.savefig('speed_plot.png')


"""
Looking to see whether there is a correlation between my average running speed,
duration or distance, and weather. Hard-coded weather values in the CSV
for temperature in °C.
"""



print("Success!")