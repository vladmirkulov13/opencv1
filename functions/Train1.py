import cv2
import numpy as np

cap = cv2.VideoCapture("../videos/Crossroads Traffic Light_Trim.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    roi1 = frame[0:400, 0:400]
    mask = object_detector.apply(roi1)
    _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Mask", mask)
    cv2.imshow("Mask1", mask1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break
