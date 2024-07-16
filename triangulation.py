import numpy as np
import cv2

class Triangulation:
    def projection_matrix(self, K, R1, t1, R2, t2):
        # self.T1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.T1 = np.hstack((R1, t1.reshape(-1, 1)))
        self.P1 = K @ self.T1

        self.T2 = np.hstack((R2, t2.reshape(-1, 1)))
        self.P2 = K @ self.T2

    def triangulate_point(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        A = np.array([
            x1 * self.P1[2, :] - self.P1[0, :],
            y1 * self.P1[2, :] - self.P1[1, :],
            x2 * self.P2[2, :] - self.P2[0, :],
            y2 * self.P2[2, :] - self.P2[1, :]])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        return X[:3]

    def triangulate_points(self, K, R1, t1, R2, t2, pts1, pts2):
        self.projection_matrix(K, R1, t1, R2, t2)
        points_3d = np.zeros((pts1.shape[0], 3))
        for index, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            points_3d[index, :] = self.triangulate_point(pt1, pt2)
        return points_3d

    def reproject_point(self, X, P):
        X_h = np.append(X, 1)
        p = P @ X_h
        p /= p[2]
        return p[:2]

    def reprojection_error(self, points3d, pts2, T2, K) ->tuple:
        '''
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        '''
        R2 = T2[:3, :3]
        t2_vec = T2[:3, 3]
        R2_vec, _ = cv2.Rodrigues(R2)
        image_points_calc, _ = cv2.projectPoints(points3d, R2_vec, t2_vec, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(pts2))
        return total_error / len(image_points_calc)
