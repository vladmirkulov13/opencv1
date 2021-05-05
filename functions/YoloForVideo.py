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


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ' Conf: ' + str(round(confidence * 100, 3))
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


may_classes = {1, 2, 3, 5, 7}
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
    # помещаем blob в сеть
    net.setInput(blob)
    # запускаем логический вывод по сети
    # и собираем прогнозы из выходных слоев
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.8
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
            if confidence > 0.2 and class_id in may_classes:
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
    while i < len(coordsForLines):
        cv2.line(frame, (coordsForLines[i][0], coordsForLines[i][1]),
                 (coordsForLines[i + 1][0], coordsForLines[i + 1][1]), (0, 0, 0))
        i += 2
    carsOnFrame = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

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
