import cv2
import numpy as np
import matplotlib.pyplot as plt

# def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
#     label = str(classes[class_id]) + ' Conf: ' + str(round(confidence * 100, 3))
#     # label = str(classes[class_id])
#     color = COLORS[class_id]
#
#     cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
#
#     cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#
# def get_output_layers(net):
#     layer_names = net.getLayerNames()
#
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
#     return output_layers
#
#
# may_classes = {1, 2, 3, 5, 7}
#
# image = cv2.imread('../photos/cars&truck&bus.jpg')
#
# Width = image.shape[1]
# Height = image.shape[0]
# scale = 0.00392
#
# classes = None
# # чтение классов из файла и запись в массив (простой список) classes
# with open('../yoloModel/yolov3.txt', 'r') as f:
#     classes = [line.strip() for line in f.readlines()]
#
# COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# # считываются файлы весов и конфигурации, создается сеть
# # на основе обученной модели yolo3
# net = cv2.dnn.readNet('../yoloModel/yolov3.weights', '../yoloModel/yolov3.cfg')
# # blob - подготвленное входное изображение для обработки моделью
# blob = cv2.dnn.blobFromImage(image, scale, (288, 288), (0, 0, 0), True, crop=False)
# # помещаем blob в сеть
# net.setInput(blob)
# # запускаем логический вывод по сети
# # и собираем прогнозы из выходных слоев
# outs = net.forward(get_output_layers(net))
#
# class_ids = []
# confidences = []
# boxes = []
# conf_threshold = 0.8
# nms_threshold = 0.4
# output = []
#
# # для каждого обнаружения из каждого выходного слоя
# # получить достоверность, идентификатор класса, параметры ограничивающей рамки
# # и игнорировать слабые обнаружения (достоверность <0,5)
# for out in outs:
#     for detection in out:
#         # первые пять элементов - это координаты объекта - х, у, ширина, выотса
#         scores = detection[5:]
#         # элементы 5-84 отвечают за увереннсоть в выборе класса
#         # найденный номер наибольшего элемента - это класс объекта
#         class_id = np.argmax(scores)
#         # сам наибольший элемент - уверенность в правильности выбора
#         confidence = scores[class_id]
#         # если уверенность в объекте > 0.5
#         # то извлекаются координаты его расположения
#         if confidence > 0.8 and class_id in may_classes:
#             center_x = int(detection[0] * Width)
#             center_y = int(detection[1] * Height)
#             w = int(detection[2] * Width)
#             h = int(detection[3] * Height)
#             x = center_x - w / 2
#             y = center_y - h / 2
#             class_ids.append(class_id)
#             confidences.append(float(confidence))
#             boxes.append([x, y, w, h])
# # подавление немаксимумов
# indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
# # отрисовка прямоугольников с учетом подавления немаксимальных
# for i in indices:
#     i = i[0]
#     box = boxes[i]
#     x = box[0]
#     y = box[1]
#     w = box[2]
#     h = box[3]
#     # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
#     output.append(classes[class_ids[i]])
#     print(classes[class_ids[i]])
#
#
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# img = cv2.imread('image17.jpg')
#
# im = cv2.resize(img, (500, 500))
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# im = cv2.filter2D(im, -1, kernel)
# vid = cv2.VideoCapture("C://Users//student//PycharmProjects//opencv1//videos//perekrestok_Trim")
# cap = cv2.VideoCapture("../videos/perekrestok_Trim.mp4")
#
# # while True:
# #     ret, frame = vid.read()
# #     cv2.imshow("Sharpening", frame)
# #     if cv2.waitKey(60) == 27:
# #         break
# while True:
#     ret, frame = cap.read()
#     cv2.imshow("a", frame)
#     if cv2.waitKey(60) == 27:
#         break

config_file = "../yoloModel/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "../yoloModel/frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)


classLabels = []
file = "../yoloModel/Labes"
with open(file, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# print(len(classLabels))

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean(127.5)
model.setInputSwapRB(True)

img = cv2.imread("../photos/first_frame.jpg")
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    if conf > 0.5:
        cv2.rectangle(img, boxes, (255, 0, 0), 2)
        cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                    color=(0, 255, 0),
                    thickness=3)
cv2.imwrite('12345.jpg', img)

cap = cv2.VideoCapture("../videos/highway_Trim.mp4")
if not cap.isOpened():
    raise IOError("Canot open video")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd == 3:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale,
                            color=(0, 255, 0),
                            thickness=3)
            cv2.imshow('Detect', frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
