from sklearn.datasets import make_blobs
from ml_algorithms.kmeans import KMeans # Custom KMeans implementation
import numpy as np

# Generate synthetic 2D classification data with 3 centers
X, y = make_blobs(
    centers=3,           # Number of centers/clusters
    n_samples=500,       # Total number of data points
    n_features=2,        # Number of features
    shuffle=True,        # Shuffle the samples
    random_state=40      # For reproducibility
)

# Output the shape of the generated data
print(X.shape)  # Should print (500, 2)

# Calculate the number of unique clusters in the true labels
clusters = len(np.unique(y))
print(f'Number of Clusters: {clusters}')

# Initialize custom KMeans algorithm
k = KMeans(k=clusters, max_iters=150, plot_steps=True, random_seed=42)

# Fit the model and get predicted cluster labels
y_pred = k.predict(X)

# Plot the final clustering result
k.plot()
