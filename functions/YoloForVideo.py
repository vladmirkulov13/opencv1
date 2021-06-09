import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
a, b = 0, 0
coordsForLines = []


def draw_markers(event, x, y, flags, param):
    global ix, iy, a, b, drawing, frame, frame_copy
    if flags == cv2.EVENT_FLAG_LBUTTON:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            frame_copy = frame.copy()
            coordsForLines.append((ix, iy))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                frame = cv2.line(
                    frame_copy.copy(), (ix, iy), (x, y), (0, 255, 0), 2)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        a, b = x, y
        coordsForLines.append((a, b))


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, centr_x, centr_y):
    # label = str(classes[class_id]) + str(centr_x) + ", " + str(centr_y)  # ' Conf: ' + str(round(confidence * 100, 3))
    if class_id == 2:
        # BGR
        color = (0, 255, 255)
    elif class_id == 7:
        color = (255, 0, 0)
    elif class_id == 5:
        color = (255, 255, 0)
    else:
        color = (0, 0, 0)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


may_classes = {1, 2, 3, 5, 7}
with open('../yoloModel/yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture("../videos/4k.mp4")

frameCount = 0
counterOfCounts = []
previousCarsCount = 0
prev_cars, prev_trucks, prev_buses, prev_bikes = [], [], [], []

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
    for i in range(2):
        counterOfCounts.append(0)
net = cv2.dnn.readNet('../yoloModel/yolov3.weights', '../yoloModel/yolov3.cfg')

while True:

    ret, frame = cap.read()

    if not ret:
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392
    cars, trucks, buses, bikes = [], [], [], []
    # blob - подготвленное входное изображение для обработки моделью
    blob = cv2.dnn.blobFromImage(frame, scale, (288, 288), (0, 0, 0), True, crop=False)
    # помещаем blob в сеть
    net.setInput(blob)
    # запускаем логический вывод по сети
    # и собираем прогнозы из выходных слоев
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.4

    # для каждого обнаружения из каждого выходного слоя
    # получить достоверность, идентификатор класса, параметры ограничивающей рамки
    # и игнорировать слабые обнаружения (достоверность <0,5)
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
            if class_id in may_classes:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # подавление немаксимумов
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # отрисовка прямоугольников с учетом подавления немаксимальных
    i = 0
    while i < len(coordsForLines):
        cv2.line(frame, (coordsForLines[i][0], coordsForLines[i][1]),
                 (coordsForLines[i + 1][0], coordsForLines[i + 1][1]), (0, 0, 0))
        i += 2

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x, y, w, h = round(x), round(y), round(w), round(h)
        centr_x = (x * 2 + w) / 2
        centr_y = (y * 2 + h) / 2
        draw_prediction(frame, class_ids[i], confidences[i], x, y, x + w, y + h, centr_x, centr_y)

        # car 2
        # motorbike 3
        # bus 5
        # truck 7
        for q in range(0, len(coordsForLines), 2):
            ix, iy, a, b = coordsForLines[q][0], coordsForLines[q][1], coordsForLines[q + 1][0], \
                           coordsForLines[q + 1][1]
            if Width - a < int(Width / 4):
                a = Width
            # CAR
            if ix <= centr_x <= a and iy - 20 <= centr_y <= b + 20 and class_ids[i] == 2:
                cv2.line(frame, (coordsForLines[q][0], coordsForLines[q][1]),
                         (coordsForLines[q + 1][0], coordsForLines[q + 1][1]), (255, 255, 255))

                counterOfCounts[q] += 1
                cars.append((centr_x, centr_y))
                for pc in prev_cars:
                    if abs(centr_x - pc[0]) < 50 and abs(centr_y - pc[1]) < 50:
                        counterOfCounts[q] -= 1

                prev_cars = cars

            cv2.putText(frame, "Cars: " + str(counterOfCounts[q]), (int((a + ix) / 2), 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)
            if len(cars) == 0:
                prev_cars.clear()
            # TRUCK
            if ix <= centr_x <= a and iy - 20 <= centr_y <= b + 20 and class_ids[i] == 7:
                cv2.line(frame, (coordsForLines[q][0], coordsForLines[q][1]),
                         (coordsForLines[q + 1][0], coordsForLines[q + 1][1]), (0, 255, 255))
                counterOfCounts[q + 1] += 1
                trucks.append((centr_x, centr_y))
                for pt in prev_trucks:
                    if abs(centr_x - pt[0]) < 100 and abs(centr_y - pt[1]) < 100:
                        counterOfCounts[q + 1] -= 1

                prev_trucks = trucks
            cv2.putText(frame, "Truck: " + str(counterOfCounts[q + 1]), (int((a + ix) / 2) + 70, 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)
            # BUS
            if ix <= centr_x <= a and iy - 20 <= centr_y <= b + 20 and class_ids[i] == 5:
                cv2.line(frame, (coordsForLines[q][0], coordsForLines[q][1]),
                         (coordsForLines[q + 1][0], coordsForLines[q + 1][1]), (0, 255, 255))
                counterOfCounts[q + 2] += 1
                buses.append((centr_x, centr_y))
                for bs in prev_buses:
                    if abs(centr_x - bs[0]) < 100 and abs(centr_y - bs[1]) < 100:
                        counterOfCounts[q + 2] -= 1

                prev_buses = buses
            cv2.putText(frame, "Bus: " + str(counterOfCounts[q + 2]), (int((a + ix) / 2) + 140, 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)
            # BIKES
            if ix <= centr_x <= a and iy - 30 <= centr_y <= b + 30 and class_ids[i] == 3:
                cv2.line(frame, (coordsForLines[q][0], coordsForLines[q][1]),
                         (coordsForLines[q + 1][0], coordsForLines[q + 1][1]), (0, 255, 255))
                counterOfCounts[q + 3] += 1
                bikes.append((centr_x, centr_y))
                for bk in prev_bikes:
                    if abs(centr_x - bk[0]) < 100 and abs(centr_y - bk[1]) < 100:
                        counterOfCounts[q + 3] -= 1

                prev_bikes = bikes
            cv2.putText(frame, "Bikes: " + str(counterOfCounts[q + 3]), (int((a + ix) / 2) + 210, 100 + 20 * q),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)
    cv2.imshow("Frame", frame)
    previousFrameCount = frameCount
    frameCount += 5
    cap.set(1, frameCount)
    if cv2.waitKey(30) == 27:
        break
