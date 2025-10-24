# -----------------------------------------------
# 🧠 K-Means Clustering on Mall Customers Dataset
# -----------------------------------------------

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 2: Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("✅ Dataset loaded successfully!")
print("\nFirst 5 rows:\n", df.head())

# Step 3: Select relevant features (Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Determine the optimal number of clusters using Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Step 5: Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o', linestyle='--', color='b')
plt.title("Elbow Method to Find Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()

# Step 6: Train KMeans with the chosen number of clusters (for example, k=5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Step 7: Display cluster centers
print("\nCluster Centers (Income, Spending Score):")
print(kmeans.cluster_centers_)

# Step 8: Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='bright',
    data=df,
    s=100
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# Step 9: Analyze the clusters
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\n📊 Cluster Summary:\n", cluster_summary)

# Step 10: Save results
df.to_csv("Clustered_Customers.csv", index=False)
print("\n✅ 'Clustered_Customers.csv' file created successfully with cluster labels!")
