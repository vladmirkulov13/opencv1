import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# ЗАХВАТ ЛИЦА И РАБОТА С НИМ В ВИДЕОПОТОКЕ
# capture = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
#
# def blur_face(img):
#     (h, w) = img.shape[:2]
#     # чем больше дельта тем размытее прямоугольник
#     # если дельты равны ширине и высоте то размытие полное
#     dW = int(w / 5.0)
#     dH = int(h / 5.0)
#     # дельты могут быть только нечетные
#     if dH % 2 == 0:
#         dH -= 1
#     if dW % 2 == 0:
#         dW -= 1
#
#     return cv2.GaussianBlur(img, (dW, dH), 0)
#
#
# while True:
#     ret, img = capture.read()
#     # scaleFactor определяет размер прямоугольника - чем больше параметр, тем больше захват
#     faces = face_cascade.detectMultiScale(img, scaleFactor=10, minNeighbors=5, minSize=(20, 20))
#
#     for (x, y, w, h) in faces:
#         # прямоугольная рамка вокруг лица
#        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         img[y:y + h, x: x + w] = blur_face(img[y:y + h, x: x + w])
#
#     cv2.imshow('From camera', img)
#
#     k = cv2.waitKey(30) & 0XFF
#     if k == 27:
#         break
#
# capture.release()
# cv2.destroyAllWindows()
# ------------------------------------------------------------------
# КЛЮЧЕВЫЕ ТОЧКИ НА ФОТО И СРАВНЕНИЕ С ДРУГОЙ
# img1 = cv2.imread('one_panda.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('46563.png', cv2.IMREAD_GRAYSCALE)
#
# # sift = cv2.SIFT_create()
#
# # kp_sift = sift.detect(img, None)
# # алгоритм нахождения ключевых точек и их дескрипторов (описание точки и окружности)
# orb = cv2.ORB_create()
# # ключевые точки и дескрипторы
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# # img = cv2.drawKeypoints(img, kp_sift, None)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
#
# matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# # cv2.imshow("Image1", img1)
# # cv2.imshow("Image2", img2)
# cv2.imshow("Result", matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ----------------------------------------------------
# ПОЛУЧЕНИЕ МАСКИ ОБЪЕКТА И ВЫДЕЛЕНИЕ ПО НЕЙ ФОНА
# original_image = cv2.imread("all.jpg")
# # насыщенные цвета
# hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
# roi_image = cv2.imread("cut.jpg")
# hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
#
# hue, saturation, value = cv2.split(hsv_roi)
# # in HSV height is 0...179
# # saturation 1...255 - насыщенность
# # make a histogram of roi1 im
# # calcHist(images, channels, mask, histSize, ranges)
# roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
# # calcBackProject(images, channels, hist, ranges,scale)
# mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
#
# # Filter for improving the noise
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# mask = cv2.filter2D(mask, -1, kernel)
# _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
#
# mask = cv2.merge((mask, mask, mask))
# result = cv2.bitwise_and(original_image, mask)
#
# cv2.imshow("Orig", original_image)
# cv2.imshow("Res", result)
# # # cv2.imshow("Orig2", hsv_original)
# # cv2.imshow("Roi", roi_image)
# # cv2.imshow("Mask", mask)
# cv2.waitKey()
# cv2.destroyAllWindows()

# plt.imshow(roi_hist)
#
# plt.show()
# ----------------------------------------------------
# ДЕТЕКТИРОВАНИЕ ОБЪЕКТА С ВЫЧИТАНИЕМ ФОНА В МАСКЕ
# РАБОТАЕТ С ЛЮТЫМ КРИНЖОМ


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ' Conf: ' + str(round(confidence * 100, 3))
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


video = cv2.VideoCapture("C://Users//пользователь//Desktop//видео//2.mp4")

_, first_frame = video.read()
cv2.imwrite("first_frame.jpg", first_frame)
image = cv2.imread('first_frame.jpg')

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None
# чтение классов из файла и запись в массив (простой список) classes
with open('../yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# считываются файлы весов и конфигурации, создается сеть
# на основе обученной модели yolo3
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# blob - подготвленное входное изображение для обработки моделью
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
# помещаем blob в сеть
net.setInput(blob)
# запускаем логический вывод по сети
# и собираем прогнозы из выходных слоев
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.9
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
        if confidence > 0.9 and class_id == 2:
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

cv2.imshow("detect", image)
x, y = int(x), int(y)
# вырезаем прямоугольник из первого кадра
roi = first_frame[y:y + h, x:x + w]
# приводим изображение к насыщенным оттенкам
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
# cv2.imshow("ROI", roi1)
# cv2.waitKey(60)
# time.sleep(10)


while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    _, truck_window = cv2.meanShift(mask, (x, y, w, h), term_criteria)
    x, y, w, h = truck_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    key = cv2.waitKey(60)
    if key == 27:
        break

# ----------------------------------------------------
