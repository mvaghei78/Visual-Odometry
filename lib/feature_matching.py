import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureMatching:
    """
    A class for performing feature matching between two images using ORB or SIFT detectors.

    Attributes
    ----------
    img1 : np.ndarray
        The first image for feature matching.
    img2 : np.ndarray
        The second image for feature matching.
    pts1 : np.ndarray
        The matched feature points in the first image.
    pts2 : np.ndarray
        The matched feature points in the second image.
    show : bool
        Whether to display the matched features visually.

    Methods
    -------
    BruteForceMatchingORB: Matches features using ORB detector and Brute-Force Matcher.
    BruteForceMatchingSIFT: Matches features using SIFT detector and Brute-Force Matcher with ratio test.
    """

    def __init__(self, img1: np.ndarray, img2: np.ndarray, show: bool = False) -> None:
        """
        Initializes the FeatureMatching class with two images.

        Parameters
        ----------
        :param np.ndarray img1: The first image for feature matching.
        :param np.ndarray img2: The second image for feature matching.
        :param bool(optional) show: If True, displays the matched features visually (default is False).
        """
        self.img1 = img1
        self.img2 = img2
        self.pts1 = None
        self.pts2 = None
        self.show = show

    def BruteForceMatchingORB(self) -> tuple:
        """
        Matches features between the two images using ORB (Oriented FAST and Rotated BRIEF) and Brute-Force Matcher.

        Returns
        -------
        :return: A tuple containing two elements:
            - pts1: np.ndarray of shape (N, 2), feature points in the first image.
            - pts2: np.ndarray of shape (N, 2), feature points in the second image.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        # Create ORB detector
        orb = cv2.ORB_create()

        # Detect ORB keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(self.img1, None)
        kp2, des2 = orb.detectAndCompute(self.img2, None)

        # Match descriptors using Brute-Force Matcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance
        matches = np.array(matches).reshape(-1, )  # Convert matches to a numpy array

        # Display matches if specified
        if self.show:
            img3 = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3), plt.show()

        # Extract matched points
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return self.pts1[:, 0, :], self.pts2[:, 0, :]

    def BruteForceMatchingSIFT(self) -> tuple:
        """
        Matches features between the two images using SIFT (Scale-Invariant Feature Transform) and Brute-Force Matcher with ratio test.

        Returns
        -------
        :return: A tuple containing two elements:
            - pts1: np.ndarray of shape (N, 2), feature points in the first image.
            - pts2: np.ndarray of shape (N, 2), feature points in the second image.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # Detect SIFT keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)

        # Match descriptors using Brute-Force Matcher with k-nearest neighbors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter good matches
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # Display matches if specified
        if self.show:
            img3 = cv2.drawMatchesKnn(self.img1, kp1, self.img2, kp2, good, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3), plt.show()

        # Extract matched points
        good = np.array(good).reshape(-1, )
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self.pts1[:, 0, :], self.pts2[:, 0, :]
