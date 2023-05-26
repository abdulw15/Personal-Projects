# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the EM-DAT dataset into a Pandas DataFrame
df = pd.read_csv('disaster_data.csv')

# Clean the data by removing missing values and formatting the columns as needed
df.dropna(inplace=True)
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])
df['Total Damage (USD)'] = df['Total Damage (US$)'].str.replace(',', '').astype(int)

# Perform exploratory data analysis to identify trends and patterns in natural disasters over time and across different regions of the world
# Aggregate the data by year and disaster type
disasters_by_year = df.groupby(df['Start Date'].dt.year)['Disaster Type'].value_counts().unstack().fillna(0)

# Visualize the data using a stacked bar chart
disasters_by_year.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Natural Disasters by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Use machine learning techniques such as regression or time series analysis to build predictive models that can forecast future natural disasters based on historical data and other factors such as climate and population trends
# Build a linear regression model to predict the economic damage caused by natural disasters
X = df[['Start Date', 'Total Deaths', 'Total Affected']]
y = df['Total Damage (USD)']
model = LinearRegression()
model.fit(X, y)

# Evaluate the performance of the predictive models using metrics such as mean absolute error or root mean squared error
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)

# Interpret the results of the analysis and use the insights gained to make recommendations for policymakers and disaster response agencies
# For example, based on the linear regression model, we could identify the factors that have the greatest impact on the economic damage caused by natural disasters and use this information to prioritize disaster preparedness and response efforts.

