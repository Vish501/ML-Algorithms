import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:
    
    def __init__(self, k: int=5, max_iters: int=100, plot_steps: bool=False, random_seed: int=42):
        """
        Initialize the KMeans clustering algorithm.
        Parameters:
        - k (int): Number of clusters to form.
        - max_iters (int): Maximum number of iterations for the algorithm to run.
        - plot_steps (bool): Whether to visually plot steps.
        - random_seed (int): Seed for reproducibility.
        """

        np.random.seed(random_seed)

        self.k = k                                # Number of clusters
        self.max_iters = max_iters                # Max iterations to prevent infinite loops
        self.plot_steps = plot_steps              # Optional plotting flag (useful for visualization)
        self.clusters = [[] for _ in range(self.k)]  # List to hold sample indices for each cluster
        self.centroids = []                       # List to hold centroid coordinates for each cluster


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run KMeans clustering and assign each sample to a cluster.

        Parameters:
        - X (np.ndarray): Data to be clustered.
        Returns:
        - labels (np.ndarray): Cluster index for each data point.
        """

        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initializing centroids by randomly selecting k samples
        rand_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in rand_idxs]

        # Optimize clusters
        for iteration in tqdm(range(self.max_iters), desc="Clustering"):
            # Assign samples to closest centroids (i.e., create clusters)
            self.clusters = self._create_clusters()

            # Plot clustering progress
            if self.plot_steps and iteration % 50 == 0:
                if self.n_features <= 3:
                    self.plot()
                else:
                    print("Cannot plot, too many features.")

            # Recalculate centroids from the current clusters
            old_centroids = self.centroids
            self.centroids = self._calculate_centroids()

            # Check for convergence (no change in centroids)
            # If the old_centroids == new_centroids stop the iterations
            if self._has_converged(old_centroids, self.centroids):
                print(f"Converged after {iteration} iterations")
                break

        # Classify the samples as the idx of their clusters
        return self._get_cluster_labels()
    

    def _closest_centroid(self, sample: np.ndarray) -> int:
        """Return the index of the nearest centroid for a given sample."""
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        return int(np.argmin(distances))


    def _create_clusters(self) -> list:
        """Assign each sample to the nearest centroid."""
        clusters = [[] for _ in range(self.k)]
        for sample_idx, sample in enumerate(self.X):
            closest_idx = self._closest_centroid(sample)
            clusters[closest_idx].append(sample_idx)
        return clusters


    def _calculate_centroids(self) -> list:
        """Compute the new centroids as the mean of the assigned samples."""
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster:  # Avoid division by zero 
                centroids[cluster_idx] = np.mean(self.X[cluster], axis=0)
        return centroids


    def _get_cluster_labels(self) -> np.ndarray:
        """Return an array of cluster labels for each sample."""
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


    def _has_converged(self, old_centroids: list, new_centroids: list) -> bool:
        """Check whether centroids have stopped changing (convergence)."""
        distances = [euclidean_distance(old, new) for old, new in zip(old_centroids, new_centroids)]
        return all(distance == 0 for distance in distances)


    def plot(self):
        """Visualize the current clustering state (2D or 3D only)."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster_indexs in self.clusters:
            points = self.X[cluster_indexs].T
            ax.scatter(*points)

        for centroid in self.centroids:
            ax.scatter(*centroid, marker="x", color="black", linewidth=2)

        plt.title("KMeans Clustering")
        plt.show()
