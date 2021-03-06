import cv2
import numpy as np

cap = cv2.VideoCapture("../videos/perekrestok_Trim.mp4")

# object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# yellow = (0, 255, 255)
#
while True:
    ret, frame = cap.read()
    # 1 - height, 2 - width
    roi1 = frame[200:500 , 0:200]
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

# mass1 = [1, 2, 3, 4, 5, 6, 7, 5, 5, 5, 5, 76, 8, 7, 6,456,  5,444, 5, 67, 4, 8, 4]
mass1 = [[1, (12, 14)], [1, (15, 17)], [1, (16, 18)], [1, (-300, -600)], [1, (-3400, -6002)], [1, (-300, -600)],
         [1, (-300, -600)], [1, (-656500, -656500)], [1, (16, 18)]]


def removeBigDiff(mass):
    i = 0
    while i < len(mass) - 1:

        if mass[i] != 0:
            if abs(mass[i + 1][1][0] - mass[i][1][0]) > 100 and abs(mass[i + 1][1][1] - mass[i][1][1]) > 100:
                del mass[i + 1]

            else:
                i += 1
    return mass


def changeArrays(masses):
    i = 0
    while i < len(masses):
        x = masses[i][-1][1][0]
        y = masses[i][-1][1][1]
        j = 0
        while j < len(masses):
            if j!=i:
                x_n = masses[j][0][1][0]
                y_n = masses[j][0][1][1]
                if abs(x_n - x) < 20 and abs(y_n - y) < 20:
                    masses[i].append(masses[j])
                    del masses[j]
                else:
                    j += 1
            else:
                continue
        i += 1

# print(mass1)
# mass = removeBigDiff(mass1)
# print(mass)

# print(mass)
