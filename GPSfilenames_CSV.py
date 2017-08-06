"""
CMPT 318 Final Project
Author: Vasundhara Gautam

GPSfilenames_CSV.py is a script that goes through the current directory
to find all the .gpx files in it and writes this list to a .csv file
with timestamps. Run this if you want to try my analysis code out with
your own .gpx files.
"""

import os
import pandas as pd

running_data = []
for filename in os.listdir(os.getcwd()):
    if (filename[-4::] == '.gpx'):
        running_data.append(filename)
        
running_df = pd.DataFrame(data=running_data, columns=['filename'])
running_df['datetime'] = running_df['filename'].str.slice(start=8, stop=23)

# Writing to GPSdata.csv and trying to avoid encoding problems

running_df.to_csv('GPSdata.csv', encoding='utf-8', index=False)
