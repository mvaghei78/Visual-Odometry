import numpy as np
import matplotlib.pyplot as plt
import os, glob
from feature_matching import FeatureMatching
from essential_matrix import EssentialMatrix
import config
from triangulation import Triangulation
from show_3d import Show_In_3D
from pnp import PnP

class VisualOdometry:
    def __init__(self, images_folder, K):
        self.read_frames(images_folder)
        self.K = K
        self.initial_R = np.eye(3)
        self.initial_t = np.zeros((3, 1))
        self.total_R = [self.initial_R]
        self.total_t = [self.initial_t]
        self.total_points = np.empty((1, 3))
        self.total_colors = np.empty((1, 3))
        self.total_pose = self.K.ravel()
        self.total_error = np.empty(0)
        self.pnp = PnP()
        self.triangulate = Triangulation()
        self.em = EssentialMatrix()


    # def update_pose(self, R, t, R_prev, t_prev):
    #     R_new = R @ R_prev
    #     t_new = R @ t_prev + t
    #     return R_new, t_new
    def to_ply(self, path, point_cloud, colors) -> None:
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

    def read_frames(self, images_folder):
        pth = os.path.join(images_folder, '*.JPG')
        # images_path = [img for img in glob.glob(pth)]
        images_path = ['./GustavIIAdolf/DSC_0351.JPG', './GustavIIAdolf/DSC_0352.JPG', './GustavIIAdolf/DSC_0353.JPG',
                       './GustavIIAdolf/DSC_0354.JPG', './GustavIIAdolf/DSC_0355.JPG', './GustavIIAdolf/DSC_0356.JPG']
        self.images = [plt.imread(f_path) for f_path in glob.glob(pth)]

    def show_3d(self, all_points_3d):
        # Flatten the list of all 3D points into a single numpy array
        all_points_3d = np.vstack(all_points_3d)

        # Convert the camera poses to a trajectory array
        trajectory = np.hstack(self.total_pose).reshape(-1, 3)

        # Show the 3D points and trajectory
        show3d = Show_In_3D(all_points_3d, trajectory)
        show3d.show_3d()

    def common_points(self, pts1, pts2, pts3) -> tuple:
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


    def vo_initialization(self):
        img1 = self.images[0]
        img2 = self.images[1]
        fm = FeatureMatching(img1, img2, show=False)
        # Feature matching between frame1 and frame2
        pts1, pts2 = fm.BruteForceMatchingSIFT()
        # Essential matrix computation between frame1 and frame2
        E, em_mask = self.em.find_essential_matrix(pts1, pts2, self.K)

        pts1 = pts1[em_mask.ravel() == 1]
        pts2 = pts2[em_mask.ravel() == 1]

        R2, t2, em_mask = self.em.decompose_essential_matrix(E, pts1, pts2, self.K)

        pts1 = pts1[em_mask.ravel() > 0]
        pts2 = pts2[em_mask.ravel() > 0]

        # Triangulation between img1 and img2
        points3d = self.triangulate.triangulate_points(self.K, self.initial_R, self.initial_t, R2, t2, pts1, pts2)
        error = self.triangulate.reprojection_error(points3d, pts2, self.triangulate.T2, self.K)
        # ideally error < 1
        print("REPROJECTION ERROR: ", error)
        R_matrix, tran_vector, pts1, pts2, points3d = self.pnp.compute_pose_with_PnP(points3d, pts1, pts2, self.K,
                                            np.zeros((5, 1), dtype=np.float32))
        self.total_pose = np.hstack((self.total_pose, self.triangulate.P1.ravel()))
        self.total_pose = np.hstack((self.total_pose, self.triangulate.P2.ravel()))
        self.total_error = np.hstack((self.total_error, error))
        return img1, img2, pts1, pts2, points3d, self.initial_R, self.initial_t, R2, t2

    def __call__(self):
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
            points3d = self.triangulate.triangulate_points(self.K, R2, t2, R3, t3, common_mask_12, common_mask_21)
            error = self.triangulate.reprojection_error(points3d, common_mask_21, T3, self.K,)
            print("Reprojection Error (After triangulation using R,t in PnP): ", error)
            self.total_pose = np.hstack((self.total_pose, self.triangulate.P2.ravel()))
            self.total_points = np.vstack((self.total_points, points3d))
            img_pxls = np.array(common_mask_21, dtype=np.int32)
            color_vector = np.array([img3[p[1], p[0]] for p in img_pxls])
            self.total_colors = np.vstack((self.total_colors, color_vector))
            self.total_error = np.hstack((self.total_error, error))
            img2 = np.copy(img3)
            pts1 = np.copy(pts2_cur)
            pts2 = np.copy(pts3)
            R1 = np.copy(R2)
            t1 = np.copy(t2)
            R2 = np.copy(R3)
            t2 = np.copy(t3)
        frame_num = np.arange(1, self.total_error.shape[0]+1)
        plt.scatter(x=frame_num, y=self.total_error)
        plt.show()
        self.to_ply("./plyfile.ply", self.total_points, self.total_colors)

if __name__ == '__main__':
    vo = VisualOdometry('.\GustavIIAdolf', config.gaustav_K)
    vo()
