import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans as SklearnKMeans

from ml_algorithms.kmeans import KMeans # Custom KMeans implementation

# Setting random seeds to make sure tests are consistent and replicable
np.random.seed(42)


# ---------- Data Generators ---------- #

def blob_data(n_clusters=3, n_samples=300, n_features=2):
    '''Generate synthetic classification data'''
    X, y = make_blobs(
        centers=n_clusters,    # Number of centers/clusters
        n_samples=n_samples,   # Total number of data points
        n_features=n_features, # Number of features
        shuffle=True,          # Shuffle the samples
        random_state=40        # For reproducibility
    )

    return X, y

def noisy_blob_data(n_clusters=3, n_samples=300, n_features=2, noise_samples=20, noise_scale=3.0, labeled_noise=True):
    """
    Generates synthetic clustered data with added noise (outliers).

    Parameters:
    - n_samples (int): Number of clustered data points.
    - n_features (int): Dimensionality of data.
    - centers (int): Number of cluster centers.
    - noise_samples (int): Number of noisy/outlier points to add.
    - noise_scale (float): Multiplier for noise spread beyond cluster range.
    - labeled_noise (bool): If True, labels noise as -1.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_combined (np.ndarray): Combined data with noise.
    - y_combined (np.ndarray): Labels, with noise labeled as -1 if labeled_noise=True.
    """
    # Generate clustered data
    X, y = blob_data(n_clusters, n_samples, n_features)

    # Calculate data bounds of the generated data
    min_X = X.min(axis=0)
    max_X = X.max(axis=0)

    # Expanding bounds by noise_scale to place outliers away from clusters
    noise_lower = min_X - noise_scale * (max_X - min_X)
    noise_upper = max_X + noise_scale * (max_X - min_X)

    # Generating uniform noise
    noise = np.random.uniform(low=noise_lower, high=noise_upper, size=(noise_samples, n_features))

    # Stacking the noise
    X_noisy = np.vstack([X, noise])
    y_noisy = np.hstack([y, -1 * np.ones(noise_samples )]) if labeled_noise else y

    return X_noisy, y_noisy


# ---------- Tests ---------- #

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


def test_kmeans_convergence(n_clusters: int=3, n_samples: int=300, n_features: int=2, max_iters: int=150, plot_steps: bool=False):
    """Tests that centroids are not null."""
    X, y = blob_data(n_clusters, n_samples, n_features)

    # Ensure the data has been properly generated
    assert len(np.unique(y)) == n_clusters

    model = KMeans(k=n_clusters, max_iters=max_iters, plot_steps=plot_steps, random_seed=42)

    # Ensure centroids are empty on initialization
    assert model.centroids == []

    model.predict(X)
    final_centroids = model.centroids

    # After prediction, centroids must be populated
    assert len(final_centroids) == n_clusters
    assert not np.isnan(final_centroids).any()
 

def test_kmeans_against_sklearn(n_clusters: int=3, n_samples: int=300, n_features: int=2, max_iters: int=150, plot_steps: bool=False):
    """Compares clustering labels against sklearn using Adjusted Rand Index."""
    X, y = blob_data(n_clusters, n_samples, n_features)

    # Ensure the data has been properly generated
    assert len(np.unique(y)) == n_clusters

    # Initializing the models
    custom_model = KMeans(k=n_clusters, max_iters=max_iters, plot_steps=plot_steps, random_seed=42)
    sklearn_model = SklearnKMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    # Getting the predictions from both models
    y_pred_custom = custom_model.predict(X)
    y_pred_sklearn = sklearn_model.fit_predict(X)

    # Finding Adjusted Rand Score by comparing the two labelings, independent of actual cluster values
    score = adjusted_rand_score(y_pred_sklearn, y_pred_custom)

    print(score)
    assert score > 0.7 # threshold for similarity


if __name__ == "__main__":

    try:
        for clusters in [2,3,6]:
            for features in[2,5,8]:
                print(f'Count -> Cluster: {clusters}, Features: {features}')
                test_kmeans_cluster_count(n_clusters=clusters, n_features=features, plot_steps=False)

                print(f'Range -> Cluster: {clusters}, Features: {features}')
                test_kmeans_label_range(n_clusters=clusters, n_features=features, plot_steps=False)

                print(f'Convergence -> Cluster: {clusters}, Features: {features}')
                test_kmeans_convergence(n_clusters=clusters, n_features=features, plot_steps=False)

                print(f'Sklearn -> Cluster: {clusters}, Features: {features}')
                test_kmeans_against_sklearn(n_clusters=clusters, n_features=features, plot_steps=False)

        print("All Tests Passed")
    except:
        print("Test Failed")
