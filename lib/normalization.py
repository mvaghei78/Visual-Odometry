import numpy as np


class Normalization:
    """
    This class provides methods for normalizing points in 2D space using a normalization matrix.

    Methods:
        normalization_matrix: Computes the normalization matrix for a set of 2D points.
        normalize_points: Normalizes a set of 2D points using the normalization matrix.
    """

    def normalization_matrix(self, pts: np.ndarray) -> np.ndarray:
        """
        Computes the normalization matrix for a set of 2D points.

        Parameters
        ----------
        :param np.ndarray pts: Array of 2D points (shape: [n_points, 2]).

        Returns
        -------
        :return: The normalization matrix.
        :rtype: np.ndarray
        """
        pts = pts.reshape(-1, 2)
        std = np.std(pts)
        mean = np.mean(pts, axis=0)
        scale = np.sqrt(2) / std
        T = np.array([
            [scale, 0, -scale * mean[0]],
            [0, scale, -scale * mean[1]],
            [0, 0, 1]
        ])
        return T

    def normalize_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Normalizes a set of 2D points using the normalization matrix.

        Parameters
        ----------
        :param np.ndarray pts: Array of 2D points (shape: [n_points, 1, 2]).

        Returns
        -------
        :return: The normalized 2D points.
        :rtype: np.ndarray
        """
        T = self.normalization_matrix(pts)
        pts = pts.reshape(-1, 2)
        homogeneous_pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
        normalized_pts = (homogeneous_pts @ T.T).T
        normalized_pts /= normalized_pts[2]
        return normalized_pts[:2].T.reshape(-1, 1, 2)
