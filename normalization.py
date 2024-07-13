import numpy as np


class Normalization:

    def normalize_points(self, points):
        points = points.reshape(-1, 2)
        # Normalization matrix
        std = np.std(points)
        mean = np.mean(points, axis=0)
        T = np.array([
            [std / np.sqrt(2), 0, mean[0]],
            [0, std / np.sqrt(2), mean[1]],
            [0, 0, 1]
        ])
        T = np.linalg.inv(T)
        homogeneous = np.ones((points.shape[0], 1))
        points = np.hstack((points, homogeneous))
        points = (points @ T).T
        points = points / points[2]
        points = points[:2].T.reshape(-1, 1, 2)
        return points
