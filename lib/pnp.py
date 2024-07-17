import cv2
import numpy as np


class PnP:
    """
    This class provides a method to compute the pose of the camera using the Perspective-n-Point (PnP) algorithm
    with the RANSAC scheme to handle outliers.

    Methods
    -------
    compute_pose_with_PnP: Computes the pose of an object from 3D-2D point correspondences using the RANSAC scheme.
    """

    def compute_pose_with_PnP(self, points3d: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray,
                              distortion_coefficients: np.ndarray) -> tuple:
        """
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

        Parameters
        ----------
        :param np.ndarray points3d: Array of 3D points (shape: [n_points, 3]).
        :param np.ndarray pts1: Array of 2D points in the first image (shape: [n_points, 2]).
        :param np.ndarray pts2: Array of 2D points in the second image (shape: [n_points, 2]).
        :param np.ndarray K: Intrinsic matrix of the camera (shape: [3, 3]).
        :param np.ndarray distortion_coefficients: Distortion coefficients of the camera (shape: [5]).

        Returns
        -------
        :return: Tuple containing:
            - R_matrix: Rotational matrix (shape: [3, 3]).
            - tran_vector: Translational matrix (shape: [3, 1]).
            - pts1: Inlier 2D points in the first image (shape: [n_inliers, 2]).
            - pts2: Inlier 2D points in the second image (shape: [n_inliers, 2]).
            - points3d: Inlier 3D points (shape: [n_inliers, 3]).
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        # Solve PnP with RANSAC to find rotation and translation vectors
        _, R_vec_calc, tran_vector, inlier = cv2.solvePnPRansac(points3d, pts2, K, distortion_coefficients,
                                                                      cv2.SOLVEPNP_ITERATIVE)

        # Convert rotation vector to rotation matrix
        R_matrix, _ = cv2.Rodrigues(R_vec_calc)

        if inlier is not None:
            points3d = points3d[inlier[:, 0]]
            pts1 = pts1[inlier[:, 0]]
            pts2 = pts2[inlier[:, 0]]

        return R_matrix, tran_vector, pts1, pts2, points3d

    # def compute_pose_with_PnP(self, points3d, points2d, K, distortion_coefficients, R_vector, initial) -> tuple:
    #     '''
    #     Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
    #     returns rotational matrix, translational matrix, image points, object points, rotational vector
    #     '''
    #     if initial == 1:
    #         points3d = points3d[:, 0, :]
    #         points2d = points2d.T
    #         R_vector = R_vector.T
    #     _, R_vec_calc, tran_vector, inlier = cv2.solvePnPRansac(points3d, points2d, K, distortion_coefficients,
    #                                                                  cv2.SOLVEPNP_ITERATIVE)
    #     # Converts a rotation vector to rotation matrix
    #     R_matrix, _ = cv2.Rodrigues(R_vec_calc)
    #
    #     if inlier is not None:
    #         points2d = points2d[inlier[:, 0]]
    #         points3d = points3d[inlier[:, 0]]
    #         R_vector = R_vector[inlier[:, 0]]
    #     return R_matrix, tran_vector, points2d, points3d, R_vector
