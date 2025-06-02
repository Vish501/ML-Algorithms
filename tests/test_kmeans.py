import pytest
import numpy as np
from sklearn.datasets import make_blobs
from ml_algorithms.kmeans import KMeans # Custom KMeans implementation

def blob_data(n_clusters, n_samples, n_features):
    '''Generate synthetic 2D classification data with 3 centers'''
    X, y = make_blobs(
        centers=n_clusters,    # Number of centers/clusters
        n_samples=n_samples,   # Total number of data points
        n_features=n_features, # Number of features
        shuffle=True,          # Shuffle the samples
        random_state=40        # For reproducibility
    )

    return X, y

def test_kmeans_cluster_count(n_clusters: int=3, n_samples: int=300, n_features: int=2, max_iters: int=150, plot_steps: bool=False):
    '''Test if KMeans can cluster data into the correct number of clusters'''
    X, y = blob_data(n_clusters, n_samples, n_features)

    # Ensure the data has been properly generated
    assert len(np.unique(y)) == n_clusters

    model = KMeans(k=n_clusters, max_iters=max_iters, plot_steps=plot_steps, random_seed=42)
    y_pred = model.predict(X)

    # Ensure the output contains the same number of values as samples provided
    assert len(y_pred) == X.shape[0]

    # Ensure the total number of clusters is the same as n_clusters
    assert len(np.unique(y_pred)) == n_clusters


def test_kmeans_label_range(n_clusters: int=3, n_samples: int=300, n_features: int=2, max_iters: int=150, plot_steps: bool=False):
    '''Test if cluster labels are valid (integers in range)'''
    X, y = blob_data(n_clusters, n_samples, n_features)

    # Ensure the data has been properly generated
    assert len(np.unique(y)) == n_clusters

    model = KMeans(k=n_clusters, max_iters=max_iters, plot_steps=plot_steps, random_seed=42)
    y_pred = model.predict(X)

    # Ensure all predicted labels are in the expacted range
    assert all(label in range(n_clusters) for label in np.unique(y_pred))


if __name__ == "__main__":

    try:
        for clusters in [2,3,6]:
            for features in[2,5,8]:
                test_kmeans_cluster_count(n_clusters=clusters, n_features=features)
                test_kmeans_label_range(n_clusters=clusters, n_features=features)

        print("All Tests Passed")
    except:
        print("Test Failed")
