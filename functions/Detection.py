import cv2
import numpy as np


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ' Conf: ' + str(round(confidence * 100, 3))
    # label = str(classes[class_id])
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


def classify(image):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # чтение классов из файла и запись в массив (простой список) classes

    # считываются файлы весов и конфигурации, создается сеть
    # на основе обученной модели yolo3
    net = cv2.dnn.readNet('../yoloModel/yolov3.weights', '../yoloModel/yolov3.cfg')
    # blob - подготвленное входное изображение для обработки моделью
    blob = cv2.dnn.blobFromImage(image, scale, (288, 288), (0, 0, 0), True, crop=False)
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
            if confidence > 0.8 and class_id in may_classes:
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
        types_on_image.append(classes[class_ids[i]])
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return types_on_image


# image = cv2.imread("../photos/cars&truck&bus.jpg")
# types = classify(image)

# for i in types:
#     print(i)
# print(len(types))
