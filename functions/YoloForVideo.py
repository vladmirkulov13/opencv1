
<<<<<<< HEAD
import math
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np
from cv2 import cv2
from tkinter import *
import sys
from tracker2 import EuclideanDistTracker

dict = {}
id = 0
drawing = False
interrupted = False
ix, iy = -1, -1
a, b = 0, 0
coordsForLines = []
# BGR
white = (255, 255, 255)
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
salat = (255, 255, 0)
pink = (255, 0, 255)

root = Tk()


def close():
    root.destroy()


def win_for_exceptions(text):
    label = Label(height=3)
    label['text'] = text
    label.pack()
    Button(text='ОК', command=close).pack()
    root.mainloop()


# def countersView(counter):
#     root = Tk()
#     label = Label(height=3)
#     label['text'] = "Cars: " + str(counter[0]) + ", Trucks: " + str(counter[3]) + ", Buses: " + str(
#         counter[2]) + ", Bikes: " + str(counter[1])
#     label.pack()
#     Button(text='ОК', command=close).pack()
#     root.mainloop()

=======
import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
a, b = 0, 0
coordsForLines = []


>>>>>>> origin/master
def draw_markers(event, x, y, flags, param):
    global ix, iy, a, b, drawing, frame, frame_copy
    if flags == cv2.EVENT_FLAG_LBUTTON:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            frame_copy = frame.copy()
<<<<<<< HEAD
            coordsForLines.append(ix)
            coordsForLines.append(iy)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                frame = cv2.line(
                    frame_copy.copy(), (ix, iy), (x, y), green, 2)
=======
            coordsForLines.append((ix, iy))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                frame = cv2.line(
                    frame_copy.copy(), (ix, iy), (x, y), (0, 255, 0), 2)
>>>>>>> origin/master


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        a, b = x, y
<<<<<<< HEAD
        coordsForLines.append(a)
        coordsForLines.append(b)
        print(coordsForLines)


def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    # color в RGB
    # car
    if class_id == 2:
        color = yellow
    # bike
    elif class_id == 3:
        color = pink
    # bus
    elif class_id == 5:
        color = salat
    # truck
    elif class_id == 7:
        color = blue
    else:
        color = black

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # cv2.putText(img, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # cv2.circle(img, (cx, cy), 2, black, 4)
=======
        coordsForLines.append((a, b))


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ' Conf: ' + str(round(confidence * 100, 3))
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
>>>>>>> origin/master


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


may_classes = {1, 2, 3, 5, 7}
<<<<<<< HEAD
try:
    net = cv2.dnn.readNet('../yoloModel/yolov3.weights', '../yoloModel/yolov3.cfg')
except:
    win_for_exceptions("Can't read the Net!")
    sys.exit()
filename = askopenfilename()
cap = cv2.VideoCapture(filename)

frameCount = 0
counterOfCounts = []
prev_cars, prev_trucks, prev_buses, prev_bikes = [], [], [], []
try:
    ret, frame = cap.read()
    frame_copy = frame.copy()
except:
    win_for_exceptions("Can't open VideoFile!")
    sys.exit()
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_markers)
print(cap.get(cv2.CAP_PROP_FPS))

while True:
    cv2.imshow('Frame', frame)
    for c in coordsForLines:
        counterOfCounts.append(0)
    if cv2.waitKey(30) == 27:
        break

while True:
    # if len(dict) > 60:
    #     # deletingIter = 1
    #     for deletingIter in range(1, 10, 1):
    #         dict.pop(deletingIter)

    detections = []
    ret, frame = cap.read()

    if not ret:
        break

    if cv2.waitKey(30) == 27:
        coordsForLines.clear()
        # for c in coordsForLines:
        #     counterOfCounts.append(0)
        # frame_copy = frame.copy()
        cv2.setMouseCallback('Frame', draw_markers)
        while True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(30) == 27:
                break

    Height = frame.shape[0]
    Width = frame.shape[1]
    scale = 0.00392
    cars, bikes, buses, trucks = [], [], [], []
    sameCar, sameBike, sameBus, sameTruck, sameObject = False, False, False, False, False
    # blob - подготвленное входное изображение для обработки моделью
    blob = cv2.dnn.blobFromImage(frame, scale, (288, 288), black, True, crop=False)
=======
with open('../yoloModel/yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
types_on_image = []

cap = cv2.VideoCapture("../videos/Road traffic video for object recognition_Trim.mp4")
# cap = cv2.VideoCapture("../videos/perekrestok_Trim.mp4")
# cap = cv2.VideoCapture("../videos/video_from_TIM.mp4")

truckCount = 0
frameCount = 0
counterOfCounts = []
previousCounterOfCounts = []
dicFrameCount = {}


cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
ret, frame = cap.read()
frame_copy = frame.copy()
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_markers)

while True:
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == 27:
        break
for c in coordsForLines:
    counterOfCounts.append(0)
    previousCounterOfCounts.append(0)


# truckOnPrevFrame = 0
while True:

    ret, frame = cap.read()
    if not ret:
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392
    i = 0

    truckOnFrame = 0

    # чтение классов из файла и запись в массив (простой список) classes

    # считываются файлы весов и конфигурации, создается сеть
    # на основе обученной модели yolo3
    net = cv2.dnn.readNet('../yoloModel/yolov3.weights', '../yoloModel/yolov3.cfg')

    # blob - подготвленное входное изображение для обработки моделью
    blob = cv2.dnn.blobFromImage(frame, scale, (288, 288), (0, 0, 0), True, crop=False)
>>>>>>> origin/master
    # помещаем blob в сеть
    net.setInput(blob)
    # запускаем логический вывод по сети
    # и собираем прогнозы из выходных слоев
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
<<<<<<< HEAD
    conf_threshold = 0.4
=======
    conf_threshold = 0.8
>>>>>>> origin/master
    nms_threshold = 0.4

    # для каждого обнаружения из каждого выходного слоя
    # получить достоверность, идентификатор класса, параметры ограничивающей рамки
<<<<<<< HEAD
=======
    # и игнорировать слабые обнаружения (достоверность <0,5)
>>>>>>> origin/master
    for out in outs:
        for detection in out:
            # первые пять элементов - это координаты объекта - х, у, ширина, выотса
            scores = detection[5:]
            # элементы 5-84 отвечают за увереннсоть в выборе класса
            # найденный номер наибольшего элемента - это класс объекта
            class_id = np.argmax(scores)
            # сам наибольший элемент - уверенность в правильности выбора
            confidence = scores[class_id]
            # если уверенность в объекте > 0.5
            # то извлекаются координаты его расположения
<<<<<<< HEAD
            if class_id in may_classes:
=======
            if confidence > 0.2 and class_id in may_classes:
>>>>>>> origin/master
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
<<<<<<< HEAD

    # подавление немаксимумов (обязательно для избегания повторов распознавания одного и того же объекта)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    try:
        for i in range(0, len(coordsForLines), 4):
            cv2.line(frame, (coordsForLines[i], coordsForLines[i + 1]),
                     (coordsForLines[i + 2], coordsForLines[i + 3]), black)
    except:
        win_for_exceptions("Don't touch!")
    for i in indices:
        sameObject = False
        now_id = 0
        i = i[0]
        # выбираем из boxes наилучшие согласно indexes
=======
    # подавление немаксимумов
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # отрисовка прямоугольников с учетом подавления немаксимальных
    while i < len(coordsForLines):
        cv2.line(frame, (coordsForLines[i][0], coordsForLines[i][1]),
                 (coordsForLines[i + 1][0], coordsForLines[i + 1][1]), (0, 0, 0))
        i += 2
    carsOnFrame = 0
    for i in indices:
        i = i[0]
>>>>>>> origin/master
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
<<<<<<< HEAD
        x, y, w, h = round(x), round(y), round(w), round(h)
        draw_prediction(frame, class_ids[i], x, y, x + w, y + h)
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        if frameCount == 0:
            id += 1
            dict[id] = (cx, cy)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
        else:
            for centres in dict.values():
                now_id += 1
                if math.hypot(cx - centres[0], cy - centres[1]) < 50:
                    sameObject = True
                    dict[now_id] = (cx, cy)
                    cv2.putText(frame, str(now_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    break
            if not sameObject:
                dict[len(dict) + 1] = (cx, cy)
                cv2.putText(frame, str(len(dict) + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        # car 2, motorbike 3, bus 5, truck 7
        for q in range(0, len(coordsForLines), 4):
            ix, iy, a, b = coordsForLines[q], coordsForLines[q + 1], coordsForLines[q + 2], \
                           coordsForLines[q + 3]
            if ix > a:
                ix, a = a, ix
            if iy > b:
                iy, b = b, iy
            if abs(ix - a) > abs(iy - b):
                # CAR
                if ix <= cx <= a and iy <= cy <= iy + 20:
                    # CAR
                    if class_ids[i] == 2:
                        cars.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), white)
                        for pc in prev_cars:
                            if abs(cx - pc[0]) < 30 and abs(cy - pc[1]) < 30:
                                sameCar = True
                        if not sameCar:
                            counterOfCounts[q] += 1
                    # BIKES
                    if class_ids[i] == 3:
                        bikes.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), pink)
                        for bk in prev_bikes:
                            if abs(cx - bk[0]) < 30 and abs(cy - bk[1]) < 30:
                                sameBike = True
                        if not sameBike:
                            counterOfCounts[q + 1] += 1
                    # BUS
                    if class_ids[i] == 5:
                        buses.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), salat)
                        for bs in prev_buses:
                            if abs(cx - bs[0]) < 30 and abs(cy - bs[1]) < 30:
                                sameBus = True
                        if not sameBus:
                            counterOfCounts[q + 2] += 1
                    # TRUCK
                    if class_ids[i] == 7:
                        trucks.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), blue)
                        for pt in prev_trucks:
                            if abs(cx - pt[0]) < 30 and abs(cy - pt[1]) < 30:
                                sameTruck = True
                        if not sameTruck:
                            counterOfCounts[q + 3] += 1
            else:
                if ix >= cx >= ix - 20 and iy <= cy <= b:
                    # CAR
                    if class_ids[i] == 2:
                        cars.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), white)
                        for pc in prev_cars:
                            if abs(cx - pc[0]) < 30 and abs(cy - pc[1]) < 30:
                                sameCar = True
                        if not sameCar:
                            counterOfCounts[q] += 1
                    # BIKES
                    if class_ids[i] == 3:
                        bikes.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), pink)
                        for bk in prev_bikes:
                            if abs(cx - bk[0]) < 30 and abs(cy - bk[1]) < 30:
                                sameBike = True
                        if not sameBike:
                            counterOfCounts[q + 1] += 1
                    # BUS
                    if class_ids[i] == 5:
                        buses.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), salat)
                        for bs in prev_buses:
                            if abs(cx - bs[0]) < 30 and abs(cy - bs[1]) < 30:
                                sameBus = True
                        if not sameBus:
                            counterOfCounts[q + 2] += 1
                    # TRUCK
                    if class_ids[i] == 7:
                        trucks.append((cx, cy))
                        cv2.line(frame, (coordsForLines[q], coordsForLines[q + 1]),
                                 (coordsForLines[q + 2], coordsForLines[q + 3]), blue)
                        for pt in prev_trucks:
                            if abs(cx - pt[0]) < 30 and abs(cy - pt[1]) < 30:
                                sameTruck = True
                        if not sameTruck:
                            counterOfCounts[q + 3] += 1

            cv2.putText(frame, "Cars: " + str(counterOfCounts[q]), (int((a + ix) / 2), 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        black, 1)
            # color = (frame[100 + 20 * q, int((a + ix) / 2) + 210][0], frame[100 + 20 * q, int((a + ix) / 2)][1],
            #          frame[100 + 20 * q, int((a + ix) / 2)][2])
            cv2.putText(frame, "Bikes: " + str(counterOfCounts[q + 1]), (int((a + ix) / 2) + 210, 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        black, 1)
            cv2.putText(frame, "Bus: " + str(counterOfCounts[q + 2]), (int((a + ix) / 2) + 140, 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        black, 1)
            cv2.putText(frame, "Truck: " + str(counterOfCounts[q + 3]), (int((a + ix) / 2) + 70, 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        black, 1)
            # number = 0
            # for d in dict.values():
            #     number += 1
            #     cv2.putText(frame, str(number), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 (255, 255, 255), 1)

    cars_copy, bikes_copy, buses_copy, trucks_copy = cars.copy(), bikes.copy(), buses.copy(), trucks.copy()
    prev_cars, prev_bikes, prev_buses, prev_trucks = cars_copy, bikes_copy, buses_copy, trucks_copy
    cars.clear()
    bikes.clear()
    buses.clear()
    trucks.clear()

    cv2.imshow("Frame", frame)
    frameCount += 5
    cap.set(1, frameCount)
    # if cv2.waitKey(1) & 0xFF == ord('1'):
    #     interrupted = True
    #     break

cap.release()
cv2.destroyAllWindows()
# if interrupted:
#     win_for_exceptions("Interrupted by user!")
# else:
win_for_exceptions("The end!")

=======

        types_on_image.append(classes[class_ids[i]])
        x, y, w, h = round(x), round(y), round(w), round(h)
        # image = frame[y:y + h, x:x + w]
        draw_prediction(frame, class_ids[i], confidences[i], x, y, x + w, y + h)
        centr_x = (x * 2 + w) / 2
        centr_y = (y * 2 + h) / 2
        prevCentr_x = 0
        prevCentr_y = 0


        # car 2
        # motorbike 3
        # bus 5
        # truck 7
        for q in range(0, len(coordsForLines), 2):
            ix, iy, a, b = coordsForLines[q][0], coordsForLines[q][1], coordsForLines[q + 1][0], coordsForLines[q + 1][
                1]
            if ix <= centr_x <= a and iy - 5 <= centr_y <= b + 5 and class_ids[i] == 2:
                cv2.line(frame, (coordsForLines[q][0], coordsForLines[q][1]),
                         (coordsForLines[q + 1][0], coordsForLines[q + 1][1]), (255, 255, 255))
                counterOfCounts[q] += 1
                carsOnFrame += 1
                if abs(centr_x - prevCentr_x) < 200 and abs(centr_y - prevCentr_y) < 200:
                    counterOfCounts[q] -= 1
                prevCentr_x, prevCentr_y = centr_x, centr_y
            cv2.putText(frame, "Cars: " + str(counterOfCounts[q]), (int((a + ix) / 2), 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 1)
            if ix <= centr_x <= a and iy - 10 <= centr_y <= b + 10 and class_ids[i] == 7:
                cv2.line(frame, (coordsForLines[q][0], coordsForLines[q][1]),
                         (coordsForLines[q + 1][0], coordsForLines[q + 1][1]), (0, 255, 255))
                counterOfCounts[q + 1] += 1

            cv2.putText(frame, "Truck: " + str(counterOfCounts[q + 1]), (int((a + ix) / 2) + 140, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 1)

    dicFrameCount[frameCount] = carsOnFrame
    # print(str(carsCount))
    # for i in range(0, len(counterOfCounts), 2):
    #     if counterOfCounts[i] - previousCounterOfCounts[i] >= 3:
    #         counterOfCounts[i] = previousCounterOfCounts[i] + 3
    #     if counterOfCounts[i+1] - previousCounterOfCounts[i+1] >= 3:
    #         counterOfCounts[i+1] = previousCounterOfCounts[i+1] + 3
    #
    # for i in range(0, len(previousCounterOfCounts), 2):
    #     previousCounterOfCounts[i] = counterOfCounts[i]
    # if truckOnFrame > 1 and truckOnFrame == truckOnPrevFrame:
    #     truckCount -= truckOnFrame
    #     truckOnPrevFrame = 0
    # else:
    #     truckOnPrevFrame = truckOnFrame
    # cv2.imwrite("first_frame.jpg", image)

    cv2.imshow("Frame", frame)
    previousFrameCount = frameCount
    frameCount += 3
    cap.set(1, frameCount)
    if cv2.waitKey(30) == 27:
        break

# image = cv2.imread("../photos/cars&truck&bus.jpg")
# types = classify(image)
>>>>>>> origin/master
