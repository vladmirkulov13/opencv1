import cv2
import numpy as np


def sift_track(frame, img):
    # cv2.imwrite("car_last.jpg", img)

    sift = cv2.SIFT_create()
    kp_image, desc_image = sift.detectAndCompute(img, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # cv2.namedWindow("object", cv2.WINDOW_NORMAL)
    # cv2.imshow("object", img)

    gray_frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)

    kp_grayframe, desc_grayframe = sift.detectAndCompute(gray_frame, None)

    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_points.append(m)
    # cv2.namedWindow("Homography", cv2.WINDOW_NORMAL)
    if len(good_points) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        # Perspective transform
        h, w = img.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        #cv2.imshow("Homography", homography)
    #else:
        #cv2.imshow("Homography", gray_frame)