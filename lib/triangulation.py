import numpy as np
import cv2

class Triangulation:
    """
    This class provides methods for computing 3D points from 2D image correspondences using triangulation.

    Methods:
        projection_matrix: Computes the projection matrices for two consecutive frames.
        triangulate_point: Triangulates a single 3D point from two 2D correspondences.
        triangulate_points: Triangulates multiple 3D points from sets of 2D correspondences.
        reproject_point: Reprojects a 3D point to 2D using a given projection matrix.
        reprojection_error: Computes the reprojection error for a set of 3D points.
    """

    def projection_matrix(self, K: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> None:
        """
        Compute the projection matrices for two consecutive frames.

        Parameters
        ----------
        :param np.ndarray K: Intrinsic matrix of the camera.
        :param np.ndarray R1: Rotation matrix of the first frame.
        :param np.ndarray t1: Translation vector of the first frame.
        :param np.ndarray R2: Rotation matrix of the second frame.
        :param np.ndarray t2: Translation vector of the second frame.
        """
        self.T1 = np.hstack((R1, t1.reshape(-1, 1)))
        self.P1 = K @ self.T1

        self.T2 = np.hstack((R2, t2.reshape(-1, 1)))
        self.P2 = K @ self.T2

    def triangulate_point(self, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        """
        Triangulate a single 3D point from two 2D correspondences.

        Parameters
        ----------
        :param np.ndarray pt1: 2D point in the first image.
        :param np.ndarray pt2: 2D point in the second image.

        Returns
        -------
        :return: Triangulated 3D point.
        :rtype: np.ndarray
        """
        x1, y1 = pt1
        x2, y2 = pt2
        A = np.array([
            x1 * self.P1[2, :] - self.P1[0, :],
            y1 * self.P1[2, :] - self.P1[1, :],
            x2 * self.P2[2, :] - self.P2[0, :],
            y2 * self.P2[2, :] - self.P2[1, :]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        return X[:3]

    def triangulate_points(self, K: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Triangulate multiple 3D points from sets of 2D correspondences.

        Parameters
        ----------
        :param np.ndarray K: Intrinsic matrix of the camera.
        :param np.ndarray R1: Rotation matrix of the first frame.
        :param np.ndarray t1: Translation vector of the first frame.
        :param np.ndarray R2: Rotation matrix of the second frame.
        :param np.ndarray t2: Translation vector of the second frame.
        :param np.ndarray pts1: 2D points in the first image.
        :param np.ndarray pts2: 2D points in the second image.

        Returns
        -------
        :return: Triangulated 3D points.
        :rtype: np.ndarray
        """
        self.projection_matrix(K, R1, t1, R2, t2)
        points_3d = np.zeros((pts1.shape[0], 3))
        for index, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            points_3d[index, :] = self.triangulate_point(pt1, pt2)
        return points_3d

    def reproject_point(self, X: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Reproject a 3D point to 2D using a given projection matrix.

        Parameters
        ----------
        :param np.ndarray X: 3D point.
        :param np.ndarray P: Projection matrix.

        Returns
        -------
        :return: Reprojected 2D point.
        :rtype: np.ndarray
        """
        X_h = np.append(X, 1)
        p = P @ X_h
        p /= p[2]
        return p[:2]

    def reprojection_error(self, points3d: np.ndarray, pts2: np.ndarray, T2: np.ndarray, K: np.ndarray) -> float:
        """
        Calculates the reprojection error, i.e., the distance between the projected points and the actual points.

        Parameters
        ----------
        :param np.ndarray points3d: 3D points.
        :param np.ndarray pts2: Corresponding 2D points in the second image.
        :param np.ndarray T2: Transformation matrix (rotation and translation) for the second image.
        :param np.ndarray K: Intrinsic matrix of the camera.

        Returns
        -------
        :return: Total reprojection error.
        :rtype: float
        """
        R2 = T2[:3, :3]
        t2_vec = T2[:3, 3]
        R2_vec, _ = cv2.Rodrigues(R2)
        image_points_calc, _ = cv2.projectPoints(points3d, R2_vec, t2_vec, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(pts2))
        return total_error / len(image_points_calc)
