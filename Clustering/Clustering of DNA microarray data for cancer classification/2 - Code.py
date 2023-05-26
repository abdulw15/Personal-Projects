# Import libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('gene_expression_data.csv')

# Data preprocessing
X = data.iloc[:,1:].values # Exclude the first column which contains gene names
X = StandardScaler().fit_transform(X) # Scale the data to have zero mean and unit variance

# Dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(principal_components)

# Visualization
plt.scatter(principal_components[:,0], principal_components[:,1], c=clusters)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering of DNA Microarray Data')
plt.show()
