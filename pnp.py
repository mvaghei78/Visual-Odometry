import cv2


class PnP:
    def compute_pose_with_PnP(self, points3d, pts1, pts2, K, distortion_coefficients) -> tuple:
        '''
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        '''
        # pts1 = pts1[:, 0, :]
        # pts2 = pts2[:, 0, :]
        _, R_vec_calc, tran_vector, inlier = cv2.solvePnPRansac(points3d, pts2, K, distortion_coefficients,
                                                                     cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation vector to rotation matrix
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
