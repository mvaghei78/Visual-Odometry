import numpy as np
import matplotlib.pyplot as plt
import os, glob
from lib.feature_matching import FeatureMatching
from lib.essential_matrix import EssentialMatrix
from config import config
from lib.triangulation import Triangulation
from lib.pnp import PnP
from lib.removeOutliers import ROL
from utils.show_3d import Show_In_3D
from utils.generate_output import Generate_Output
from utils.dataset_process import DatasetProcess


class VisualOdometry:
    def __init__(self, images_folder: str, K: np.ndarray) -> None:
        self.K = K
        self.initial_R = np.eye(3)
        self.initial_t = np.zeros((1, 3))
        self.total_R = np.array(self.initial_R)
        self.total_t = np.array(self.initial_t)
        self.total_points = np.zeros((1, 3))
        self.total_colors = np.zeros((1, 3))
        self.total_pose = self.K.ravel()
        self.total_error = np.empty(0)
        self.pnp = PnP()
        self.triangulate = Triangulation()
        self.em = EssentialMatrix()
        self.rol = ROL()
        self.gen_output = Generate_Output()
        self.dataset_process = DatasetProcess()
        self.images = self.dataset_process.read_frames(images_folder)


    def common_points(self, pts1: np.ndarray, pts2: np.ndarray, pts3: np.ndarray) -> tuple:
        '''
        Finds the common points between 1 and 2 , 2 and 3
        returns common points of 1-2, common points of 2-3, mask of common points 1-2 , mask for common points 2-3
        '''
        common_pts_12 = []  # List to store indices of points in pts1 that are also found in pts2.
        common_pts_21 = []  # List to store indices of points in pts2 that are also found in pts1.
        for i in range(pts1.shape[0]):
            match = np.where(pts2 == pts1[i, :])[0]
            if match.size != 0:
                common_pts_12.append(i)
                common_pts_21.append(match[0])

        # Create masked arrays for common points between 1-2
        mask_array_img2 = np.ma.array(pts2, mask=False)
        mask_array_img2.mask[common_pts_21] = True
        mask_array_img2 = mask_array_img2.compressed()  # Keep elements that there mask value is False
        mask_array_img2 = mask_array_img2.reshape(int(mask_array_img2.shape[0] / 2), 2)

        # Create masked arrays for common points between image 2-3
        mask_array_img3 = np.ma.array(pts3, mask=False)
        mask_array_img3.mask[common_pts_21] = True
        mask_array_img3 = mask_array_img3.compressed()
        mask_array_img3 = mask_array_img3.reshape(int(mask_array_img3.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_img2.shape, mask_array_img3.shape)
        return np.array(common_pts_12), np.array(common_pts_21), mask_array_img2, mask_array_img3


    def vo_initialization(self) -> tuple:
        img1 = self.images[0]
        img2 = self.images[1]
        fm = FeatureMatching(img1, img2, show=False)
        # Feature matching between frame1 and frame2
        pts1, pts2 = fm.BruteForceMatchingSIFT()
        # Essential matrix computation between frame1 and frame2
        E, em_mask = self.em.find_essential_matrix(pts1, pts2, self.K)

        pts1, pts2 = self.rol.remove_outliers_2d(pts1, pts2, em_mask.ravel() == 1)

        R2, t2, em_mask = self.em.decompose_essential_matrix(E, pts1, pts2, self.K)

        pts1, pts2 = self.rol.remove_outliers_2d(pts1, pts2, em_mask.ravel() > 0)

        # Triangulation between img1 and img2
        points3d = self.triangulate.triangulate_points(self.K, self.initial_R, self.initial_t, R2, t2, pts1, pts2)
        error = self.triangulate.reprojection_error(points3d, pts2, self.triangulate.T2, self.K)
        # ideally error < 1
        print("REPROJECTION ERROR: ", error)
        R_matrix, tran_vector, pts1, pts2, points3d = self.pnp.compute_pose_with_PnP(points3d, pts1, pts2, self.K,
                                            np.zeros((5, 1), dtype=np.float32))

        img_pxls = np.array(pts2, dtype=np.int32)
        color_vector = np.array([img2[p[1], p[0]] for p in img_pxls])
        self.insert(P1=self.triangulate.P1, P2=self.triangulate.P2, R1=self.initial_R, R2=R2,
                    t1=self.initial_t, t2=t2, points3d=points3d, color_vector=color_vector, error=error)
        return img1, img2, pts1, pts2, points3d, self.initial_R, self.initial_t, R2, t2

    def insert(self, P1=None, P2=None, R1=None, R2=None, t1=None, t2=None, points3d=None, color_vector=None, error=None):
        if P1 is not None:
            self.total_pose = np.hstack((self.total_pose, P1.ravel()))
        if P2 is not None:
            self.total_pose = np.hstack((self.total_pose, P2.ravel()))
        if R1 is not None:
            self.total_R = np.vstack((self.total_R, R1))
        if R2 is not None:
            self.total_R = np.vstack((self.total_R, R2))
        if t1 is not None:
            self.total_t = np.vstack((self.total_t, t1.ravel()))
        if t2 is not None:
            self.total_t = np.vstack((self.total_t, t2.ravel()))
        if points3d is not None:
            self.total_points = np.vstack((self.total_points, points3d))
        if color_vector is not None:
            self.total_colors = np.vstack((self.total_colors, color_vector))
        if error is not None:
            self.total_error = np.hstack((self.total_error, error))

    def __call__(self) -> None:
        img1, img2, pts1, pts2, points3d, R1, t1, R2, t2 = self.vo_initialization()
        for i in range(2, len(self.images)):
            img3 = self.images[i]
            fm = FeatureMatching(img2, img3, show=False)
            # Feature matching between frame1 and frame2
            pts2_cur, pts3 = fm.BruteForceMatchingSIFT()
            if i != 2:
                points3d = self.triangulate.triangulate_points(self.K, R1, t1, R2, t2, pts1, pts2)

            common_pts_12, common_pts_21, common_mask_12, common_mask_21 = self.common_points(pts2, pts2_cur, pts3)
            common_pts_3 = pts3[common_pts_21]
            common_pts_cur = pts2_cur[common_pts_21]

            R3, t3, common_pts_cur, common_pts_3, points3d = self.pnp.compute_pose_with_PnP(points3d[common_pts_12], common_pts_cur, common_pts_3, self.K, np.zeros((5, 1), dtype=np.float32))
            T3 = np.hstack((R3, t3))

            error = self.triangulate.reprojection_error(points3d, common_pts_3, T3, self.K)
            print("Reprojection Error (After calculate R,t with PnP): ", error)
            points3d = self.triangulate.triangulate_points(self.K, R2, t2, R3, t3, common_pts_cur, common_pts_3)
            error = self.triangulate.reprojection_error(points3d, common_pts_3, T3, self.K)
            print("Reprojection Error (After triangulation using R,t in PnP): ", error)
            img_pxls = np.array(common_pts_3, dtype=np.int32)
            color_vector = np.array([img3[p[1], p[0]] for p in img_pxls])
            img2 = np.copy(img3)
            pts1 = np.copy(pts2_cur)
            pts2 = np.copy(pts3)
            R1 = np.copy(R2)
            t1 = np.copy(t2)
            R2 = np.copy(R3)
            t2 = np.copy(t3)
            self.insert(P2=self.triangulate.P2, R2=R2, t2=t2, points3d=points3d, color_vector=color_vector, error=error)
        frame_num = np.arange(1, self.total_error.shape[0]+1)
        plt.scatter(x=frame_num, y=self.total_error)
        plt.xlabel('Frame number')
        plt.ylabel('Error')
        plt.show()
        self.gen_output.generate_ply(config.PLY_PATH, self.total_points, self.total_colors)

        # Show the 3D points and trajectory
        show3d = Show_In_3D(self.total_points, self.total_t)
        show3d.show_3d()

if __name__ == '__main__':
    vo = VisualOdometry(config.GAUSTAV_PATH, config.GAUSTAV_K)
    vo()
