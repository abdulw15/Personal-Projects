import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv('kc_house_data.csv')

# Drop irrelevant columns
data.drop(['id', 'date'], axis=1, inplace=True)

# Convert date columns to datetime format
data['yr_built'] = pd.to_datetime(data['yr_built'], format='%Y')
data['yr_renovated'] = pd.to_datetime(data['yr_renovated'], format='%Y')

# Handle missing values
data.dropna(inplace=True)

# Handle categorical variables
data = pd.get_dummies(data, columns=['zipcode'])

# Univariate analysis
plt.figure(figsize=(20, 20))
for i, col in enumerate(data.columns):
    plt.subplot(8, 4, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Bivariate analysis
plt.figure(figsize=(20, 20))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(8, 4, i+1)
    sns.scatterplot(x=col, y='price', data=data)
plt.tight_layout()
plt.show()

# Correlation analysis
plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Create new features
data['age'] = (pd.to_datetime('today').year - data['yr_built'].dt.year).astype(int)
data['renovated'] = np.where(data['yr_renovated'] > pd.to_datetime('today').year - 5, 1, 0)
data['living_area_ratio'] = data['sqft_living'] / data['sqft_lot']

# Scale and normalize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_scaled)
data['kmeans_labels'] = kmeans.labels_

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_scaled)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# count number of data points in each cluster
cluster_counts = pd.DataFrame(kmeans.labels_, columns=['cluster']).groupby('cluster').size().reset_index(name='counts')

# plot cluster distribution
plt.figure(figsize=(10,6))
sns.barplot(x='cluster', y='counts', data=cluster_counts)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Counts')
plt.show()

# calculate mean values for each feature for each cluster
cluster_means = X_scaled.groupby(kmeans.labels_).mean()

# plot cluster means
plt.figure(figsize=(12,8))
sns.heatmap(cluster_means.T, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Cluster Profiles')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()

# reduce dimensionality of data to two dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# create DataFrame with PCA results and cluster labels
pca_df = pd.DataFrame({'x': X_pca[:,0], 'y': X_pca[:,1], 'cluster': kmeans.labels_})

# plot clusters in two dimensions
plt.figure(figsize=(10,6))
sns.scatterplot(x='x', y='y', hue='cluster', data=pca_df, palette='deep')
plt.title('Clusters in Two Dimensions')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# add cluster labels to original dataset
df['cluster'] = kmeans.labels_

# save dataset with cluster labels
df.to_csv('housing_clusters.csv', index=False)
