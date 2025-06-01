import numpy as np

class KMeans:
    
    def __init__(self, k: int=5, max_iters: int=100, plot_steps: bool=False):
        """
        Initialize the KMeans clustering algorithm.
        Parameters:
        - k (int): Number of clusters to form.
        - max_iters (int): Maximum number of iterations for the algorithm to run.
        - plot_steps (bool): Whether to visually plot steps (useful for 2D datasets).
        """        
        self.k = k                                # Number of clusters
        self.max_iters = max_iters                # Max iterations to prevent infinite loops
        self.plot_steps = plot_steps              # Optional plotting flag (useful for visualization)
        self.clusters = [[] for _ in range(self.k)]  # List to hold sample indices for each cluster
        self.centroids = []                       # List to hold centroid coordinates for each cluster

    def predict(self, X: np.ndarray):
        self.X = X
        self.n_samples, self.features = X.shape

        # Initializing centroids using random indexs from given data
        rand_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in rand_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            pass
        
    def plot(self):
        # Plot
        pass
    
    def _create_clusters(self, centroids):
        # Assign samples to the closest cluster
        pass

    def _is_converged(self, old_centroids, centroids):
        # Check if there has been no change in the centroids from the current and prior iteration
        pass

    def _get_cluster_labels(self, clusters):
        # Get the label of each sample based on the cluster they are assigned to
        pass



    