import pandas as pd

# Load dataset
spotify_data = pd.read_csv('spotify_data.csv')

# Explore the dataset
print(spotify_data.head())
print(spotify_data.info())
print(spotify_data.describe())
print(spotify_data.isnull().sum())
print(spotify_data.duplicated().sum())

from sklearn.preprocessing import StandardScaler
import pandas as pd

# Drop irrelevant features
spotify_data = spotify_data.drop(['id', 'name', 'artists'], axis=1)

# Normalize or standardize the data
scaler = StandardScaler()
spotify_data_scaled = scaler.fit_transform(spotify_data)

# Encode categorical variables
spotify_data_encoded = pd.get_dummies(spotify_data)

# Split the dataset into training and testing subsets
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(spotify_data_encoded, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of danceability
sns.histplot(data=spotify_data, x='danceability')
plt.show()

# Scatterplot of energy vs loudness
sns.scatterplot(data=spotify_data, x='energy', y='loudness')
plt.show()

# Correlation matrix
sns.heatmap(spotify_data.corr(), annot=True)
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Choose number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Train model with K-Means algorithm
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_train)

# Predict cluster labels for testing subset
y_pred = kmeans.predict(X_test)

# Evaluate model performance with silhouette score
score = silhouette_score(X_test, y_pred)
print(f"Silhouette Score: {score}")

import matplotlib.pyplot as plt

# Scatterplot of danceability vs energy, colored by cluster label
plt.scatter(X_test['danceability'], X_test['energy'], c=y_pred)
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()

# Analysis of each cluster
cluster_df = pd.concat([X_test.reset_index(drop=True), pd.Series(y_pred)], axis=1)
cluster_df.columns = list(X_test.columns) + ['Cluster Label']
for i in range(kmeans.n_clusters):
    print(f"\nCluster {i}:")
    print(cluster_df[cluster_df['Cluster Label']==i].describe())

# Additional features to enhance analysis
# - Time signature
# - Mode
# - Acousticness

# Additional models to compare performance
# - DBSCAN
# - Hierarchical clustering

# Robustness testing with external validation
# - Collect additional datasets for comparison and testing
# - Compare results with alternative clustering methods and metrics
