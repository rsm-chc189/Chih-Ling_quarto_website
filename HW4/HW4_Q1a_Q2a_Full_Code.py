# --- Question 1a ---

# Question 1a: K-Means Clustering on Palmer Penguins
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load and prepare the dataset
penguins_df = pd.read_csv("palmer_penguins.csv")
penguins_df = penguins_df[['bill_length_mm', 'flipper_length_mm']].dropna()
X = penguins_df.values

# Evaluate for K = 2 to 7
wcss = []
sil_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, labels))

# Plot WCSS and Silhouette Score
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o')
plt.title("Within-Cluster Sum of Squares")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")

plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score")

plt.tight_layout()
plt.show()


# --- Question 2a ---

# Question 2a: K-Nearest Neighbors on Synthetic Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generate training data
np.random.seed(42)
n = 100
x1_train = np.random.uniform(-3, 3, n)
x2_train = np.random.uniform(-3, 3, n)
boundary_train = np.sin(4 * x1_train) + x1_train
y_train = (x2_train > boundary_train).astype(int)

train_df = pd.DataFrame({'x1': x1_train, 'x2': x2_train, 'y': y_train})
X_train = train_df[['x1', 'x2']].values

# Generate test data
np.random.seed(2025)
x1_test = np.random.uniform(-3, 3, n)
x2_test = np.random.uniform(-3, 3, n)
boundary_test = np.sin(4 * x1_test) + x1_test
y_test = (x2_test > boundary_test).astype(int)
X_test = np.column_stack((x1_test, x2_test))

# Evaluate KNN for k = 1 to 30
accuracy_scores = []

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(range(1, 31), accuracy_scores, marker='o')
plt.title("KNN Accuracy on Test Data")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
