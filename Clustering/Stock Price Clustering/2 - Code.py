import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
stock_data = pd.read_csv('stock_data.csv')

# Viewing the first few rows of the dataset
stock_data.head()

# Checking the dimensions of the dataset
stock_data.shape

# Checking for missing values in the dataset
stock_data.isna().sum()

# Checking the statistical summary of the dataset
stock_data.describe()
# Dropping columns that aren't required
stock_data.drop(['date', 'tic'], axis=1, inplace=True)

# Filling missing values with 0
stock_data.fillna(0, inplace=True)

# Checking for missing values again
stock_data.isna().sum()

# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
stock_data_scaled = scaler.fit_transform(stock_data)
# Creating new features
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(stock_data_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Combining the new features with the original dataset
stock_data_final = pd.concat([principal_df, stock_data], axis=1)
# Scaling the data again
scaler_final = StandardScaler()
stock_data_scaled_final = scaler_final.fit_transform(stock_data_final)
# Using the elbow method to choose the number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(stock_data_scaled_final)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# Performing KMeans clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(stock_data_scaled_final)

# Adding the cluster labels to the original dataset
stock_data['cluster'] = kmeans.labels_
# Performing Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
agg_clustering.fit(stock_data_scaled_final)

# Adding the cluster labels to the original dataset
stock_data['cluster'] = agg_clustering.labels_
# Performing DBSCAN clustering
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=2)
dbscan.fit(stock_data_scaled_final)

# Adding the cluster labels to the original dataset
stock_data['cluster'] = dbscan.labels_
# Evaluating the Model

# Import the required libraries
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Compute the silhouette score for KMeans model
silhouette_avg = silhouette_score(X, y_pred)
print("The average silhouette_score is :", silhouette_avg)

# Compute the Davies-Bouldin score for KMeans model
db_score = davies_bouldin_score(X, y_pred)
print("The Davies-Bouldin score is :", db_score)
