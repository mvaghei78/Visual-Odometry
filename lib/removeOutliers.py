from sklearn.cluster import DBSCAN
import numpy as np


class ROL:
    """
    A class to remove outliers from 2D and 3D point sets using different methods.

    Methods
    -------
    remove_outliers_2d: Removes 2D outliers based on a provided condition.
    remove_outliers_3d: Removes outliers from a 3D point cloud using DBSCAN clustering.
    """

    def remove_outliers_2d(self, pts1: np.ndarray, pts2: np.ndarray, condition: bool) -> tuple:
        """
        Removes outliers from 2D point sets based on a given condition.

        Parameters
        ----------
        :param np.ndarray pts1: The first set of 2D points.
        :param np.ndarray pts2: The second set of 2D points.
        :param bool condition: A boolean array indicating which points to keep.

        Returns
        -------
        :return: A tuple containing two elements:
            - Filtered pts1: np.ndarray
            - Filtered pts2: np.ndarray
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        return pts1[condition], pts2[condition]

    def remove_outliers_3d(self, point_clouds: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """
        Removes outliers from a 3D point cloud using DBSCAN clustering algorithm.

        Parameters
        ----------
        :param np.ndarray point_clouds: An array of 3D points.
        :param np.ndarray colors: An array of colors corresponding to each 3D point.

        Returns
        -------
        :return: An array containing the inlier 3D points.
        :rtype: np.ndarray
        """
        # Perform DBSCAN clustering
        db = DBSCAN(eps=0.3, min_samples=10).fit(point_clouds)

        # Retrieve labels from the clustering
        labels = db.labels_

        # Points labeled -1 are considered outliers
        outliers = point_clouds[labels == -1]
        inliers = point_clouds[labels != -1]

        return inliers
