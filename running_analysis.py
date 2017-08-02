# CMPT 318 Final Project
# Author: Vasundhara Gautam

# running_analysis.py is the main document in this project and running it will
# produce all the beautiful graphs explained and analysed in the report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from butterworth_distance import get_distance
from scipy import stats

# Reading all GPX files in the directory is helpful for someone else to
# quickly analyze their data / for me to add more
# Also slicing the GPX filename for the date and time of the run

running_data = []
for filename in os.listdir(os.getcwd()):
    if (filename[-4::] == '.gpx'):
        running_data.append(filename)
        
running_df = pd.DataFrame(data=running_data, columns=['filename'])
running_df['datetime'] = running_df['filename'].str.slice(start=8, stop=23)

# Time stuff
# Parsing datetime strings using strptime-equivalent in pandas
# Need to plot against timestamps but matplotlib will not plot datetime objects

running_df['datetime'] = pd.to_datetime(running_df['datetime'], format='%Y-%m-%d_%H%M')
def to_timestamp(inputdatetime):
    return inputdatetime.timestamp()
running_df['timestamp'] = running_df['datetime'].apply(to_timestamp)

# Distance
running_df['filename'].apply(get_distance)