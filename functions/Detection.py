import cv2
import numpy as np

import time


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # label = str(classes[class_id]) + str(math.floor(confidence * 100)) + "%"
    # label = str(classes[class_id])
    # color = COLORS[class_id]
    if class_id == 2:
        color = (0, 0, 0)

    elif class_id == 7:
        color = (0, 255, 0)

    elif class_id == 5:
        color = (0, 0, 255)

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

net = cv2.dnn.readNet('../yoloModel/yolov3.weights', '../yoloModel/yolov3.cfg')


def classify(image):
    types_on_image = []
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # чтение классов из файла и запись в массив (простой список) classes
    start = time.time()
    # считываются файлы весов и конфигурации, создается сеть
    # на основе обученной модели yolo3

    # blob - подготвленное входное изображение для обработки моделью
    blob = cv2.dnn.blobFromImage(image, scale, (288, 288), (0, 0, 0), True, crop=False)
    # помещаем blob в сеть
    net.setInput(blob)
    # запускаем логический вывод по сети
    # и собираем прогнозы из выходных слоев
    outs = net.forward(get_output_layers(net))
    stop = time.time()
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.6
    nms_threshold = 0.4
    carCount = 0
    truckCount = 0
    fullCarConf = 0
    fullTruckConf = 0

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
            if (class_id == 2 or class_id == 7) and confidence >= 0.6:
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
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        if class_ids[i] == 2:
            fullCarConf += confidences[i]
            carCount += 1
        if class_ids[i] == 7:
            fullTruckConf += confidences[i]
            truckCount += 1
        types_on_image.append(classes[class_ids[i]])
        # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if carCount != 0:
        avgCarConf = fullCarConf / carCount
    else:
        avgCarConf = 0
    if truckCount != 0:
        avgTruckConf = fullTruckConf / truckCount
    else:
        avgTruckConf = 0
    # cv2.putText(image, "All types: " + str(carCount + truckCount), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # cv2.putText(image, "Car: " + str(carCount), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 2)
    # cv2.putText(image, "Avg conf: " + str(round(avgCarConf, 2) * 100) + "%", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 0, 0), 2)
    # cv2.putText(image, "Truck: " + str(truckCount), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.putText(image, "Avg conf: " + str(round(avgTruckConf, 2) * 100) + "%", (100, 140), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (0, 255, 0), 2)
    # cv2.putText(image, "Time: " + str(round(stop - start, 2)), (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 2)
    # cv2.imshow("Car Detecting", image)
    # cv2.waitKey(0)
    return types_on_image


# image = cv2.imread("../photos/videt-mnogo-mashin-na-doroge-vo-sne.jpg")
# types = classify(image)
