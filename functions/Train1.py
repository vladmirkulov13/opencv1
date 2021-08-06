# import cv2
# import numpy as np
# import cv2
import math
import time
import tkinter as tk, threading
import imageio
import xlwt
from PIL import Image, ImageTk

# cap = cv2.VideoCapture("../videos/highway.mp4")
# length = int(cap.get(cv2.CAP_PROP_FPS))
# print( length )
# ret, frame = cap.read()
# print(frame[100, 100])
# a = (frame[100, 100][0], frame[100, 100][1], frame[100, 100][2])
# print(a)
# frame = cv2.bitwise_not(frame)
# a = (frame[100, 100][0], frame[100, 100][1], frame[100, 100][2])
# print(a)

# def get_key(d, value):
#     for k, v in d.items():
#         if v == value:
#             return k
#
#
# dict = {1: (100, 100), 2: (200, 200), 3: (300, 300)}
#
# a = get_key(dict, (100, 100))
# print(len(dict))
# a0, b0 = 1068, 868
# a1, b1 = 1075, 886
# x = 10
# w = 50
# y = 20
# h = 30
# start = time.time()
# q = abs(a0-a1) < 30 and abs(b0-b1) < 30
# # cx = int(w / 2 + x)
# # cy = int(h / 2 + y)
# stop = time.time()
# print(round(stop - start, 10))
# start = time.time()
# q = math.hypot(a0-a1, b0-b1) < 30
# # cx = (x + x + w) // 2
# # cy = (y + y + h) // 2
# stop = time.time()
# print(round(stop - start, 10))
import xlrd
from numpy.ma import array

data = xlrd.open_workbook('../functions/Test.xls')
sheet = data.sheet_by_index(0)
row_number = sheet.nrows

dict = {}

if row_number > 0:
    for row in range(row_number):
        row1 = sheet.row_values(row)
        res = []
        # res = [c.strip() for c in row1[0].split(',') if not c.isspace()]
        # if res[0][0] == '[':
        #     res[0] = res[0][1:]
        # if res[-1][-1] == ']':
        #     res[-1] = res[-1][:-1]
        # for i in range(len(res)):
        #     if res[0] == '[':
        #         r = r[1:]
        #     elif r[-1] == "]":
        #         r = r[:-1]
        for value in row1:
            res.append(int(value))
        dict[res[0]] = res[1:]

wb = xlwt.Workbook()
ws = wb.add_sheet("Test1")
i = 0
for d in dict.keys():
    ws.write(i, 0, d)
    j = 1
    for dd in dict[d]:
        ws.write(i, j, dd)
        j += 1
    i += 1

wb.save("Test1.xls")

# See also: https://gist.github.com/bsdnoobz/8464000
# video_name = "../videos/highway.mp4" #This is your video file path
# video = imageio.get_reader(vdeo_name)
#
# def stream(label):
#
#     for image in video.iter_data():
#         frame_image = ImageTk.PhotoImage(Image.fromarray(image))
#         label.config(image=frame_image)
#         label.image = frame_image
#         # print(frame_image)
#
# if __name__ == "__main__":
#
#     root = tk.Tk()
#     my_label = tk.Label(root)
#     my_label.pack()
#     thread = threading.Thread(target=stream, args=(my_label,))
#     thread.daemon = 1
#     thread.start()
#     root.mainloop()
# while True:
#     while ret:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
from tkinter import messagebox as mb

# from tkinter.filedialog import askopenfilename
# filename = askopenfilename()


# def close():
#     root.destroy()
#
#
# a = [0, 1, 2, 3, 4, 5]
#
# root = Tk()
# label = Label(height=3)
# label['text'] = "cars: " + str(a[0]) + ", cars1: " + str(a[1]) + ", cars2: " + str(a[2]) + ", cars3: " + str(a[3])
# label.pack()
# root.mainloop()

# except:
#     print("Some other error occurred!")
# else:
#     print("All right")
# finally:
#     print("Finally")


# def check():
#     answer = mb.askyesno(
#         title="Вопрос",
#         message="Перенести данные?")
#     if answer:
#         s = entry.get()
#         entry.delete(0, END)
#         label['text'] = s


#
# cap = cv2.VideoCapture("../videos/perekrestok_Trim.mp4")
#
# # object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# # yellow = (0, 255, 255)
# #
# while True:
#     ret, frame = cap.read()
#     # 1 - height, 2 - width
#     roi1 = frame[200:500 , 0:200]
#     # mask = object_detector.apply(roi1)
#     # _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
#     # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.imshow("Mask", mask)
#     # cv2.imshow("Mask1", mask1)
#     # cv2.putText(frame, "Hello world!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
#     # cv2.imshow('result', frame)
#     cv2.imshow("Frame", frame)
#     cv2.imshow("ROI", roi1)
#     key = cv2.waitKey(30)
#     if key == 27:
#         break
# # diap = []
# # for i in range(100,500,1):
# #     diap.append(i)
# # pass
#
# # mass1 = [1, 2, 3, 4, 5, 6, 7, 5, 5, 5, 5, 76, 8, 7, 6,456,  5,444, 5, 67, 4, 8, 4]
# mass1 = [[1, (12, 14)], [1, (15, 17)], [1, (16, 18)], [1, (-300, -600)], [1, (-3400, -6002)], [1, (-300, -600)],
#          [1, (-300, -600)], [1, (-656500, -656500)], [1, (16, 18)]]
#
#
# def removeBigDiff(mass):
#     i = 0
#     while i < len(mass) - 1:
#
#         if mass[i] != 0:
#             if abs(mass[i + 1][1][0] - mass[i][1][0]) > 100 and abs(mass[i + 1][1][1] - mass[i][1][1]) > 100:
#                 del mass[i + 1]
#
#             else:
#                 i += 1
#     return mass
#
#
# def changeArrays(masses):
#     i = 0
#     while i < len(masses):
#         x = masses[i][-1][1][0]
#         y = masses[i][-1][1][1]
#         j = 0
#         while j < len(masses):
#             if j!=i:
#                 x_n = masses[j][0][1][0]
#                 y_n = masses[j][0][1][1]
#                 if abs(x_n - x) < 20 and abs(y_n - y) < 20:
#                     masses[i].append(masses[j])
#                     del masses[j]
#                 else:
#                     j += 1
#             else:
#                 continue
#         i += 1
#
# # print(mass1)
# # mass = removeBigDiff(mass1)
# # print(mass)
#
# # print(mass)
