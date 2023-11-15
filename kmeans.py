import numpy as np

class KMeansClustering:
    
    def __init__(self, k) -> None:
        self.k = k
        self.centroids = None
        self._sse_score = None

    def euclidean_distance(self, data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def __sum_of_squared_errors_calc(self, centroids, data, y):
        sum_of_errors = 0.0
        for d in data:
            sum_of_errors += np.sum((centroids[y] - d) ** 2, axis=1)

        return sum_of_errors

    def get_sum_of_squared_error(self):
        return self._sse_score
    
    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(
            low=np.amin(X, axis=0),
            high=np.amax(X, axis=0),
            size=(self.k, X.shape[1]))

        y = []
        for _ in range(max_iterations):
            y = []
            for data_point in X:

                distances = self.euclidean_distance(
                    data_point=data_point,
                    centroids=self.centroids)
                # print(distances.shape)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y = np.asarray(y)

            cluster_indices = []

            for idx in range(self.k):
                cluster_indices.append(np.argwhere(y == idx))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 1e-3:
                break
            else:
                self.centroids = np.array(cluster_centers)

        # Calculate the final SSE after performing K-means
        self._sse_score = self.__sum_of_squared_errors_calc(X, self.centroids, y)
        
        return y