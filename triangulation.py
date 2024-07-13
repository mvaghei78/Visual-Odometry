import numpy as np


class Triangulation:
    def projection_matrix(self, K, R, t):
        self.M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.M1 = K @ self.M1

        self.M2 = np.hstack((R, t.reshape(-1, 1)))
        self.M2 = K @ self.M2

    def triangulate_point(self, pt1, pt2):
        x1, y1 = pt1[0]
        x2, y2 = pt2[0]
        A = np.array([
            x1 * self.M1[2, :] - self.M1[0, :],
            y1 * self.M1[2, :] - self.M1[1, :],
            x2 * self.M2[2, :] - self.M2[0, :],
            y2 * self.M2[2, :] - self.M2[1, :]])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        return X[:3]

    def triangulate_points(self, K, R, t, pts1, pts2):
        self.projection_matrix(K, R, t)
        points_3d = np.zeros((pts1.shape[0], 3))
        for index, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            points_3d[index, :] = self.triangulate_point(pt1, pt2)
        return points_3d

    def reproject_point(self, X, P):
        X_h = np.append(X, 1)
        p = P @ X_h
        p /= p[2]
        return p[:2]

    def reprojection_error(self, K, R, t, pts1, pts2, points_3d):
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t.reshape(-1, 1)))

        errors = []
        for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            X = points_3d[i]
            p1_reproj = self.reproject_point(X, P1)
            p2_reproj = self.reproject_point(X, P2)

            error1 = np.linalg.norm(pt1 - p1_reproj)
            error2 = np.linalg.norm(pt2 - p2_reproj)
            errors.append(error1 + error2)

        average_error = np.mean(errors)
        return average_error
