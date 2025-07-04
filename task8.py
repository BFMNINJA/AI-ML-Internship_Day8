import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load and visualize dataset
df = pd.read_csv('Mall_Customers.csv')
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Optional: Visualize Age vs. Annual Income vs. Spending Score
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 2. Fit K-Means and assign cluster labels (initial K=5)
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)
df['Cluster'] = labels

# 3. Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.show()

# 4. Visualize clusters with color-coding
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1', s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clusters')
plt.legend()
plt.show()

# Optional: PCA for 2D visualization (if you use more features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='Set1')
plt.title('Clusters visualized with PCA')
plt.show()

# 5. Evaluate clustering using Silhouette Score
score = silhouette_score(X, labels)
print(f"Silhouette Score for K=5: {score:.3f}")

# Brief analysis
print("\nAnalysis:")
print("The Elbow Method plot helps to visually determine the optimal number of clusters (K).")
print("A higher Silhouette Score (closer to 1) indicates better-defined clusters.")
print("You may re-run KMeans with the optimal K found from the Elbow plot for best results.")