# a. Overview of the project
# The project aims to cluster financial time series data to identify patterns and trends in the market.

# b. Dataset description
# The dataset consists of daily stock prices of 30 companies from the S&P 500 index from 2010 to 2020.

# c. Project objectives
# The objectives of the project are to:
# - Preprocess and clean the data
# - Explore and visualize the data
# - Cluster the time series data using appropriate algorithms
# - Evaluate the clustering performance and interpret the results
# - Deploy the clustering model and evaluate its performance on new data

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a. Load the dataset
df = pd.read_csv('stock_prices.csv')

# b. Data cleaning and transformation
# Remove missing values
df.dropna(inplace=True)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set date column as index
df.set_index('Date', inplace=True)

# c. Feature engineering
# Create new columns for daily returns and percentage change
df['Returns'] = df['Close'].diff()
df['Pct_change'] = df['Close'].pct_change()

# d. Data normalization
# Normalize the data using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

# a. Summary statistics
print(df.describe())

# b. Data visualization
# Plot the daily closing prices of all companies
df.plot(figsize=(12,8), title='Daily Closing Prices')

# c. Correlation analysis
# Compute correlation matrix of daily returns
corr_matrix = df.corr(method='pearson')
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Daily Returns')
plt.show()

# a. Selecting appropriate clustering algorithms
# In this example, we will use K-means and Hierarchical clustering algorithms

# b. Feature selection and dimensionality reduction
# We will use PCA to reduce the dimensionality of the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# c. Clustering the time series data
# K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_pca)
labels_kmeans = kmeans.predict(df_pca)

# Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hierarchical = AgglomerativeClustering(n_clusters=3)
labels_hierarchical = hierarchical.fit_predict(df_pca)

# a. Determine the optimal number of clusters
# We can use the elbow method to determine the optimal number of clusters for K-means clustering
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_pca)
    wcss.append(kmeans.inertia_)

