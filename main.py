import matplotlib.pyplot as plt
from essential_matrix import EssentialMatrix
from feature_matching import FeatureMatching
from triangulation import Triangulation
from show_3d import Show_In_3D
import config

frame1 = plt.imread('./frames/img1.jpg')
frame2 = plt.imread('./frames/img2.jpg')
fm = FeatureMatching(frame1, frame2, show=False)
pts1, pts2 = fm.BruteForceMatchingORB()
em = EssentialMatrix(pts1, pts2, config.K)

E, mask = em.find_essential_matrix()
print('Essential Matrix')
print(E)
R, t, _ = em.decompose_essential_matrix(E)

pts1, pts2 = fm.remove_outliers(mask, pts1, pts2)
triangulate = Triangulation()
points_3d = triangulate.triangulate_points(config.K, R, t, pts1, pts2)
average_error = triangulate.reprojection_error(config.K, R, t, pts1, pts2, points_3d)
print(points_3d)
print(average_error)

show3d = Show_In_3D(points_3d)
show3d.show_3d()


