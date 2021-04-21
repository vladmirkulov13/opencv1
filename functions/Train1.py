# import cv2
# import numpy as np
# import argparse
#
# # cap = cv2.VideoCapture("C://Users//student//PycharmProjects//opencv1//videos//Road traffic video for object recognition.mp4")
# #
# # # object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# # # yellow = (0, 255, 255)
# # #
# # while True:
# #     ret, frame = cap.read()
# #     # 1 - height, 2 - width
# #     roi1 = frame[400:600, 100:400]
# #     roi2 = cv2.resize(frame, (300, 300))
# #
# #     # mask = object_detector.apply(roi1)
# #     # _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
# #     # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #     # cv2.imshow("Mask", mask)
# #     # cv2.imshow("Mask1", mask1)
# #     # cv2.putText(frame, "Hello world!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
# #     # cv2.imshow('result', frame)
# #     cv2.imshow("Frame", frame)
# #     cv2.imshow("ROI", roi2)
# #     key = cv2.waitKey(30)
# #     if key == 27:
# #         break
# # # diap = []
# # # for i in range(100,500,1):
# # #     diap.append(i)
# # # pass
# #
# # # mass1 = [1, 2, 3, 4, 5, 6, 7, 5, 5, 5, 5, 76, 8, 7, 6,456,  5,444, 5, 67, 4, 8, 4]
# # mass1 = [[1, (12, 14)], [1, (15, 17)], [1, (16, 18)], [1, (-300, -600)], [1, (-3400, -6002)], [1, (-300, -600)],
# #          [1, (-300, -600)], [1, (-656500, -656500)], [1, (16, 18)]]
# #
# #
# #
# #
# # def removeBigDiff(mass):
# #     i = 0
# #     while i < len(mass) - 1:
# #
# #         if mass[i] != 0:
# #             if abs(mass[i + 1][1][0] - mass[i][1][0]) > 100 and abs(mass[i + 1][1][1] - mass[i][1][1]) > 100:
# #                 del mass[i + 1]
# #
# #             else:
# #                 i += 1
# #     return mass
# #
# #
# # def changeArrays(masses):
# #     i = 0
# #     while i < len(masses):
# #         x = masses[i][-1][1][0]
# #         y = masses[i][-1][1][1]
# #         j = 0
# #         while j < len(masses):
# #             if j!=i:
# #                 x_n = masses[j][0][1][0]
# #                 y_n = masses[j][0][1][1]
# #                 if abs(x_n - x) < 20 and abs(y_n - y) < 20:
# #                     masses[i].append(masses[j])
# #                     del masses[j]
# #                 else:
# #                     j += 1
# #             else:
# #                 continue
# #         i += 1
# #
# # # print(mass1)
# # # mass = removeBigDiff(mass1)
# # # print(mass)
# #
# # # print(mass)
# import cv2 as cv
#
# drawing = False
#
# ix, iy = -1, -1
# xq, yq = -1, -1
#
# # ix, iy - входные
# # x ,y - выходные
# def draw_markers(event, x, y, flags, param):
#     global ix, iy, xq, yq, drawing, frame, frame_copy
#     if flags == cv.EVENT_FLAG_LBUTTON:
#         if event == cv.EVENT_LBUTTONDOWN:
#             drawing = True
#             ix, iy = x, y
#             frame_copy = frame.copy()
#         elif event == cv.EVENT_MOUSEMOVE:
#             if drawing:
#                 frame = cv.rectangle(
#                     frame_copy.copy(), (ix, iy), (x, y), (0, 255, 0), 2)
#             xq = x
#             yq = y
#
#         # elif event == cv.EVENT_LBUTTONUP:
#         #     print("Alt + lmouse up")
#         #     drawing = False
#         #     cv.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
#
#     # elif event == cv.EVENT_LBUTTONUP:
#     #     print("Draw crosshair")
#     #     cv.drawMarker(frame, (x, y), (255, 0, 0), 0, 16, 2, 8)
#
#
# cap = cv.VideoCapture('../videos/perekrestok_Trim.mp4')
# cap.set(cv.CAP_PROP_POS_FRAMES, 1)
# ret, frame = cap.read()
# frame_copy = frame.copy()
# cv.namedWindow('frame')
# cv.setMouseCallback('frame', draw_markers)
#
#
# while (True):
#     cv.imshow('frame', frame)
#     if cv.waitKey(60) == 27:
#         break
# print(ix, iy, xq, yq)
# if ix > xq:
#     a = ix
#     ix = xq
#     xq = a
# if iy > yq:
#     a = iy
#     iy = yq
#     yq = a
# diap_x = [i for i in range(ix, xq, 1)]
# diap_y = [i for i in range(iy, yq, 1)]
#
# print(diap_x)
# print(diap_y)
# cap.release()
# cv.destroyAllWindows()
import cv2

cap = cv2.VideoCapture("../videos/Road traffic video for object recognition.mp4")
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        count += 30  # i.e. at 30 fps, this advances one second
        cap.set(1, count)
        cv2.imshow('a', frame)
        if cv2.waitKey(30) == 27:
            break
    else:
        cap.release()
        break
