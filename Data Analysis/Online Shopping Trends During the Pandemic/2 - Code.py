import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('google_analytics_data.csv')

#Exploratory Data Analysis (EDA):

# Check the shape of the data
print("The shape of the data is:", data.shape)

# Check for missing values
print("Missing values:", data.isnull().sum())

# Check the data types of the columns
print(data.dtypes)

# Descriptive statistics
print(data.describe())

# Check the unique values in each column
for col in data.columns:
    print(col, ':', data[col].nunique())

# Visualize the distribution of target variable
sns.histplot(data['total_transactionRevenue'])
plt.show()

#Step 3 : Data Preprocessing

# Drop irrelevant columns
data.drop(['sessionId', 'visitStartTime'], axis=1, inplace=True)

# Replace missing values with 0
data.fillna(0, inplace=True)

# Convert date columns to datetime format
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# Convert categorical variables to numerical
for col in ['channelGrouping', 'deviceCategory', 'operatingSystem', 'browser', 'isMobile']:
    data[col] = pd.Categorical(data[col])
    data[col] = data[col].cat.codes


#Step 4 : Visualizations

# Visualize total revenue by date
date_revenue = data.groupby(['date']).agg({'total_transactionRevenue': 'sum'}).reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='total_transactionRevenue', data=date_revenue)
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.title('Total Revenue by Date')
plt.show()

# Visualize revenue by device category
device_revenue = data.groupby(['deviceCategory']).agg({'total_transactionRevenue': 'sum'}).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='deviceCategory', y='total_transactionRevenue', data=device_revenue)
plt.xlabel('Device Category')
plt.ylabel('Total Revenue')
plt.title('Total Revenue by Device Category')
plt.show()
