
# Question 2a: Step-by-Step KNN on Synthetic Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Generate training data
np.random.seed(42)
n = 100
x1_train = np.random.uniform(-3, 3, n)
x2_train = np.random.uniform(-3, 3, n)
boundary_train = np.sin(4 * x1_train) + x1_train
y_train = (x2_train > boundary_train).astype(int)

# Visualize training data
plt.figure()
plt.scatter(x1_train, x2_train, c=y_train, cmap='bwr', edgecolor='k')
plt.title("Training Data with Wiggly Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.show()

# Step 2: Generate test data
np.random.seed(2025)
x1_test = np.random.uniform(-3, 3, n)
x2_test = np.random.uniform(-3, 3, n)
boundary_test = np.sin(4 * x1_test) + x1_test
y_test = (x2_test > boundary_test).astype(int)

X_train = np.column_stack((x1_train, x2_train))
X_test = np.column_stack((x1_test, x2_test))

# Step 3: Implement and evaluate KNN from k=1 to 30
accuracy_scores = []

for k in range(1, 31):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)

# Step 4: Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(range(1, 31), accuracy_scores, marker='o')
plt.title("KNN Accuracy on Test Data")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
