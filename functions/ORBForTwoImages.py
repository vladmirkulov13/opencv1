import cv2
import numpy as np

img1 = cv2.imread("../photos/one_panda.jpg")
img2 = cv2.imread("../photos/alotof_pandas.jpg")
# sift = cv2.SIFT_create()
# # surf = cv2.xfeatures2d.SURF_create()
# # orb = cv2.ORB_create(nfeatures=1500)
#
# keypoints_sift, descriptors = sift.detectAndCompute(img, None)
# # keypoints_surf, descriptors = surf.detectAndCompute(img, None)
# # keypoints_orb, descriptors = orb.detectAndCompute(img, None)
#
# img = cv2.drawKeypoints(img, keypoints_sift, None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)


cv2.imshow("Im1", img1)
cv2.imshow("Im2", img2)
cv2.imshow("match", matching_result)
cv2.waitKey(0)
