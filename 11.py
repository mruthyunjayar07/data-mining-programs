import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\PC\GermanCredit.csv")

# Encode categorical attributes
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Features and target class
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Predicted cluster labels
labels = kmeans.labels_

# Cluster centroids
centroids = kmeans.cluster_centers_

# Since cluster labels may be reversed, check both mappings
accuracy1 = accuracy_score(y, labels)
accuracy2 = accuracy_score(y, 1 - labels)
final_accuracy = max(accuracy1, accuracy2)

# Print results
print("Cluster Labels:\n", labels)
print("\nCluster Centroids:\n", centroids)

# Silhouette Score
sil_score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", sil_score)

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y, labels))

# Accuracy
print("\nAccuracy:", final_accuracy * 100)

# Plot clusters (using first two features for visualization)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.title("K-Means Clustering on German Credit Dataset")
plt.show()
