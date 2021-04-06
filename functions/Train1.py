import cv2
import numpy as np

cap = cv2.VideoCapture("../videos/Crossroads Traffic Light_Trim.mp4")

# object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
yellow = (0, 255, 255)

while True:
    ret, frame = cap.read()
    # 1 - height, 2 - width
    roi1 = frame[600:1000, 1100:1700]
    # mask = object_detector.apply(roi1)
    # _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Mask1", mask1)
    # cv2.putText(frame, "Hello world!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
    # cv2.imshow('result', frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi1)
    key = cv2.waitKey(30)
    if key == 27:
        break
# diap = []
# for i in range(100,500,1):
#     diap.append(i)
# pass
