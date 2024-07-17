import numpy as np
import cv2


class EssentialMatrix:
    """
    This class provides methods for computing and decomposing the essential matrix in stereo vision.

    Methods:
        find_essential_matrix: Computes the essential matrix from corresponding points using the RANSAC algorithm.
        decompose_essential_matrix: Decomposes the essential matrix into rotation and translation using the camera intrinsic matrix.
        manual_decompose: Manually decomposes the essential matrix into rotation matrices and translation vector.
    """

    def find_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple:
        """
        Computes the essential matrix from corresponding points using the RANSAC algorithm.

        Parameters
        ----------
        :param np.ndarray pts1: Array of 2D points in the first image (shape: [n_points, 2]).
        :param np.ndarray pts2: Array of 2D points in the second image (shape: [n_points, 2]).
        :param np.ndarray K: Camera intrinsic matrix (shape: [3, 3]).

        Returns
        -------
        :return: A tuple containing the essential matrix and the mask indicating inliers.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        E /= E[2, 2]  # Normalize the essential matrix
        return E, mask

    def decompose_essential_matrix(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple:
        """
        Decomposes the essential matrix into rotation and translation matrices using the camera intrinsic matrix.

        Parameters
        ----------
        :param np.ndarray E: Essential matrix (shape: [3, 3]).
        :param np.ndarray pts1: Array of 2D points in the first image (shape: [n_points, 2]).
        :param np.ndarray pts2: Array of 2D points in the second image (shape: [n_points, 2]).
        :param np.ndarray K: Camera intrinsic matrix (shape: [3, 3]).

        Returns
        -------
        :return: A tuple containing the rotation matrix, translation vector, and the mask indicating inliers.
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray)
        """
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        t = t.reshape(-1, )
        t = t / np.linalg.norm(t)  # Normalize the translation vector
        return R, t, mask

    def manual_decompose(self, E: np.ndarray) -> tuple:
        """
        Manually decomposes the essential matrix into two possible rotation matrices and a translation vector.

        Parameters
        ----------
        :param np.ndarray E: Essential matrix (shape: [3, 3]).

        Returns
        -------
        :return: A tuple containing two rotation matrices and a translation vector.
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray)
        """
        # Decompose the Essential Matrix E
        U, S, Vt = np.linalg.svd(E)
        E_test = U @ np.diag(S) @ Vt
        print('Computed Essential Matrix from Decomposition Matrices:')
        print(E_test)

        # Correct possible sign issues
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1

        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])

        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt

        t = U[:, 2]
        return R1, R2, t
