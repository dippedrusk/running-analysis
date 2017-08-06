"""
CMPT 318 Final Project
Author: Vasundhara Gautam

running_analysis.py is the main document in this project and running it will
produce all the beautiful graphs explained and analysed in the report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from butterworth_distance import get_distance, get_time
from scipy import stats


"""
Reading all GPX files in the directory is helpful for someone else to
quickly analyze their data / for me to add more
Also slicing the GPX filename for the date and time of the run
"""

running_data = []
for filename in os.listdir(os.getcwd()):
    if (filename[-4::] == '.gpx'):
        running_data.append(filename)
        
running_df = pd.DataFrame(data=running_data, columns=['filename'])
running_df['datetime'] = running_df['filename'].str.slice(start=8, stop=23)


"""
Time stuff
Parsing datetime strings using strptime-equivalent in pandas
Need to plot against timestamps but matplotlib will not plot datetime objects
"""

running_df['datetime'] = pd.to_datetime(running_df['datetime'], format='%Y-%m-%d_%H%M')
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

print("Success!")