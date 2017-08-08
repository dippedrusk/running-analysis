"""
CMPT 318 Final Project
Author: Vasundhara Gautam

running_analysis.py is the main document in this project and running it will
produce all the beautiful graphs explained and analysed in the report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from butterworth_distance import get_distance, get_time
from scipy import stats
import seaborn as sns
sns.set(style="whitegrid")
import sys

assert sys.version_info >= (3,4)

"""
Getting the data in
Get the GPS filename list from the CSV file GPSdata.csv, parse the datetime
strings, and create timestamps to be able to do regressions
"""
print('Reading data in...\n')
inputfilename = sys.argv[1]
running_df = pd.read_csv(inputfilename, parse_dates=True, infer_datetime_format=True, encoding='utf-8')
running_df['datetime'] = pd.to_datetime(running_df['datetime']) # Redundancy to be 200% sure this works
# Timestamps are necessary to be able to do regressions
def to_timestamp(inputdatetime):
    return inputdatetime.timestamp()
running_df['timestamp'] = running_df['datetime'].apply(to_timestamp)

"""
Getting smoothed distance for all run files and doing some preliminary
statistics on this data
"""
print('Calculating distances and times...\n')
get_distance = np.vectorize(get_distance)
get_time = np.vectorize(get_time)
running_df['distance'] = get_distance(running_df['filename'])
running_df['duration'] = get_time(running_df['filename'])
running_df['avg_speed'] = running_df['distance'] / (running_df['duration']*60.0)

print('Doing linear regressions and creating plots...\n')
# Distance: Linear regression and graph with scatter plot and best-fit line
fit = stats.linregress(running_df['timestamp'], running_df['distance'])
print('The p-value of run distance over time is %.3f' % fit.pvalue)

plt.figure()
sns.set_palette(sns.light_palette('purple'))
plt.plot(running_df['datetime'], running_df['distance'], 'r.', markersize=15, label='Runs')
plt.plot(running_df['datetime'], running_df['timestamp']*fit.slope+fit.intercept, 'b-', linewidth=3, label='Best-fit line')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Run distance [m]',fontsize=17)
plt.title('Scatterplot of Running Distance',fontsize=19)
plt.xticks(rotation=40)
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
plt.savefig('distance_plot.png')

if (fit.pvalue < 0.05):
    print('At the 5% significance level, we can reject the null hypothesis that my run distance has not changed over time.')
    print('Creating a residual plot...')
    residuals = running_df['distance'] - (running_df['timestamp']*fit.slope+fit.intercept)
    plt.figure()
    sns.set_palette(sns.color_palette('muted'))
    sns.distplot(residuals)
    plt.xlabel('Difference between predicted and measured distance values',fontsize=17)
    plt.ylabel('Relative frequency',fontsize=17)
    plt.title('Graph of Residuals',fontsize=19)
    plt.tight_layout()
    plt.savefig('residuals.png')

# Duration: Linear regression and graph with just a scatter plot
fit = stats.linregress(running_df['timestamp'], running_df['duration'])
print('The p-value of run duration over time is %.3f' % fit.pvalue)
plt.figure()
plt.plot(running_df['datetime'], running_df['duration'], 'r.', markersize=15)
plt.plot(running_df['datetime'], running_df['timestamp']*fit.slope+fit.intercept, 'b-', linewidth=3, label='Best-fit line')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Run duration [minutes]',fontsize=17)
plt.title('Scatterplot of Running Duration',fontsize=19)
plt.xticks(rotation=40)
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
plt.savefig('duration_plot.png')


# Speed: Linear regression and graph with a scatter plot and best-fit line again
fit = stats.linregress(running_df['timestamp'], running_df['avg_speed'])
print('The p-value of average run speed over time is %.3f\n' % fit.pvalue)
plt.figure()
plt.plot(running_df['datetime'], running_df['avg_speed'], 'r.', markersize=15, label='Runs')
plt.plot(running_df['datetime'], running_df['timestamp']*fit.slope+fit.intercept, 'b-', linewidth=3, label='Best-fit line')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Average run speed [m/s]',fontsize=17)
plt.title('Scatterplot of Running Speed',fontsize=19)
plt.xticks(rotation=40)
plt.legend(loc=4, fontsize=14)
plt.tight_layout()
plt.savefig('speed_plot.png')


"""
Looking to see whether there is a correlation between my average running speed,
duration or distance, and weather. Hard-coded weather values in the CSV
for temperature in °C.
"""
print('Splitting the data into runs on hot days and runs on cool days...')

heat_threshold = 19.0
def isHot(data):
    if data >= heat_threshold:
        return 'Hot'
    return 'Cool'

isHot = np.vectorize(isHot)
running_df['isHot'] = isHot(running_df['temperature'])
grouped = running_df.groupby('isHot')
hotdays = grouped.get_group('Hot').reset_index(drop=True)
nothotdays = grouped.get_group('Cool').reset_index(drop=True)

# More pretty graphs
print('Drawing pretty graphs...')

plt.figure()
sns.set_palette(sns.color_palette('muted'))
sns.lmplot('temperature', 'distance', data=running_df, fit_reg=False, hue="isHot", legend=False)
plt.title('Average run distance and Temperature',fontsize=19)
plt.xlabel('Temperature [°C]',fontsize=17)
plt.ylabel('Run distance [m]',fontsize=17)
plt.legend(loc=3, fontsize=14, frameon=True)
plt.tight_layout()
plt.savefig('hotscatter.png')

sns.set_palette(sns.light_palette('green'))
sns.factorplot(x="isHot", y="distance", data=running_df, size=4, kind="bar")
plt.xlabel('Temperature [°C]',fontsize=17)
plt.ylabel('Run distance [m]',fontsize=17)
plt.savefig('hot_distance.png')

sns.factorplot(x="isHot", y="duration", data=running_df, size=4, kind="bar")
plt.xlabel('Temperature [°C]',fontsize=17)
plt.ylabel('Run duration [minutes]',fontsize=17)
plt.savefig('hot_duration.png')

sns.factorplot(x="isHot", y="avg_speed", data=running_df, size=4, kind="bar")
plt.xlabel('Temperature [°C]',fontsize=17)
plt.ylabel('Average run speed [m/s]',fontsize=17)
plt.savefig('hot_speed.png')

# Not enough data to do a normality test, so impossible to judge a t-test
# or ANOVA if I do one. Instead, a non-parametric test is used:
hotpvalue = stats.mannwhitneyu(hotdays['avg_speed'], nothotdays['avg_speed']).pvalue
print('The p-value of the Mann-Whitney U test on hot/cool run distances is %.3f\n' % hotpvalue)


print("Success!\n")