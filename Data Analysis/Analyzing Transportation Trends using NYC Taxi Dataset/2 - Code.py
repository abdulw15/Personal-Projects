import pandas as pd

# Importing the dataset
df = pd.read_csv('train.csv')

# Removing missing values
df.dropna(inplace=True)

# Removing outliers
df = df[df['trip_duration'] <= 3600]

# Converting date and time variables into appropriate format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

#Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Basic statistical analysis
print(df.describe())

# Histogram of trip duration
plt.hist(df['trip_duration'], bins=50)
plt.show()

# Box plot of trip duration by passenger count
sns.boxplot(x='passenger_count', y='trip_duration', data=df)
plt.show()

# Scatter plot of trip duration by distance
plt.scatter(df['distance'], df['trip_duration'])
plt.xlabel('Distance (miles)')
plt.ylabel('Trip Duration (seconds)')
plt.show()

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

#Step 3: Feature Engineering and Selection

import numpy as np

# Creating new features
df['distance'] = np.sqrt((df['pickup_latitude'] - df['dropoff_latitude']) ** 2 
                         + (df['pickup_longitude'] - df['dropoff_longitude']) ** 2)
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.dayofweek

# Feature selection using correlation matrix
selected_features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude', 'distance', 'pickup_hour', 'pickup_day',
                     'store_and_fwd_flag']
df = df[selected_features + ['trip_duration']]
