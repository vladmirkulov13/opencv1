import cv2
import numpy as np

# метод сравнение двух последующих кадров на предмет изменения
cap = cv2.VideoCapture(
    "C://Users//пользователь//Desktop//видео//perek_Trim.mp4")


def classify(image):
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


    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None
    # чтение классов из файла и запись в массив (простой список) classes
    with open('../yoloModel/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
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
    output = []

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
        # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        output.append(classes[class_ids[i]])

    return output


ret, frame1 = cap.read()
ret, frame2 = cap.read()
i = 0
carCount, busCount, truckCount = 0, 0, 0
while cap.isOpened():
    if ret == False:
        break
    # находим разницу между первыми двумя кадрами
    # записываем разницу в diff
    diff = cv2.absdiff(frame1, frame2)
    # переводим diff в серый цвет
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # блюрим полученное серое изображение используя матрицу 5х5, состоящую из единиц для филтра гаусса
    # и что-то типо откланения по осям = 0 - рассчитывается автоматически
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # установление порога - все пиксели больше 25 становятс белыми - остальные черными
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # размытие изображения
    dilated = cv2.dilate(thresh, None, iterations=3)
    # поиск контуров
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1500:
            continue
        # i += 1
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        outing = classify(frame1)
        if(len(outing) > 0):
            print(outing[0])
            if(outing[0] == 'car'):
                carCount += 1
            if (outing[0] == 'bus'):
                busCount += 1
            if (outing[0] == 'truckCount'):
                truckCount += 1
        # for ou in outing:
        #     cv2.putText(frame2, str(ou), (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print(outing)
        # print(i)

    # отрисовка контуров - зеленый цвет, толщина - 2
    # cv2.drawContours(frame2, contours, -1, (0, 255, 0), 2)
    # frame2 = frame2[y:y+h, x:x+w]
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

print(carCount, busCount, truckCount)
cv2.destroyAllWindows()
cap.release()
