import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate some random regression data
X, y = make_regression(n_samples=100, n_features=4, noise=0.2, random_state=42)

# Train a RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
rf.fit(X, y)

# Select the first tree from the forest
tree = rf.estimators_[0]

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=[f"Feature {i}" for i in range(X.shape[1])], filled=True)
plt.show()
