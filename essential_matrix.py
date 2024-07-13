import numpy as np
import cv2
from normalization import Normalization

class EssentialMatrix:
    def __init__(self, pts1, pts2, K):
        self.pts1 = pts1
        self.pts2 = pts2
        self.K = K

    def find_essential_matrix(self):
        E, mask = cv2.findEssentialMat(self.pts1, self.pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        E /= E[2, 2]
        return E, mask

    def decompose_essential_matrix(self, E):
        _, R, t, mask = cv2.recoverPose(E, self.pts1, self.pts2, self.K)
        t = t.reshape(-1, )
        t = t / np.linalg.norm(t)
        return R, t, mask

        # # Decompose the Essential Matrix E
        # U, S, Vt = np.linalg.svd(E)
        # E_test = U @ np.diag(S) @ Vt
        # print('Computed Essential Matrix from Decomposition Matrices:')
        # print(E_test)
        #
        # # Correct possible sign issues
        # if np.linalg.det(U) < 0:
        #     U[:, -1] *= -1
        # if np.linalg.det(Vt) < 0:
        #     Vt[-1, :] *= -1
        #
        # W = np.array([[0, -1, 0],
        #               [1, 0, 0],
        #               [0, 0, 1]])
        #
        # R1 = U @ W @ Vt
        # R2 = U @ W.T @ Vt
        #
        # t = U[:, 2]
        # print(R1)
        # print(R2)
        # print(t)
