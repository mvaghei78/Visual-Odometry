import cv2
import numpy as np
import matplotlib.pyplot as plt

class FeatureMatching:
    def __init__(self, img1, img2, show=False) -> None:
        self.img1 = img1
        self.img2 = img2
        self.pts1 = None
        self.pts2 = None
        self.show = show

    def BruteForceMatchingORB(self):
        orb = cv2.ORB_create()

        kp1, dest1 = orb.detectAndCompute(self.img1, None)
        kp2, dest2 = orb.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(dest1, dest2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = np.array(matches).reshape(-1, )
        if self.show:
            img3 = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3), plt.show()
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return self.pts1, self.pts2

    def remove_outliers(self, mask, pts1, pts2):
        inliers_pts1 = pts1[mask.ravel() == 1]
        inliers_pts2 = pts2[mask.ravel() == 1]
        return inliers_pts1, inliers_pts2

    def BruteForceMatchingSIFT(self):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        if self.show:
            img3 = cv2.drawMatchesKnn(self.img1, kp1, self.img2, kp2, good, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3), plt.show()
        good = np.array(good).reshape(-1, )
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self.pts1, self.pts2

