import numpy as np


class Normalization:
    def normalization_matrix(self, pts):
        pts = pts.reshape(-1, 2)
        # Normalization matrix
        std = np.std(pts)
        mean = np.mean(pts, axis=0)
        scale = np.sqrt(2) / std
        T = np.array([
            [scale, 0, -scale * mean[0]],
            [0, scale, -scale * mean[1]],
            [0, 0, 1]
        ])
        return T

    def normalize_points(self, pts):
        T = self.normalization_matrix(pts)
        pts = pts.reshape(-1, 2)
        homogeneous_pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
        normalized_pts = (homogeneous_pts @ T.T).T
        normalized_pts /= normalized_pts[2]
        return normalized_pts[:2].T.reshape(-1, 1, 2)
