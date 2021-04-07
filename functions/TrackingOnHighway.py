from tracker import *
from Detection import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("../videos/Crossroads Traffic Light_Trim.mp4")

# Object detection from Stable camera
# маска на основе гаусса
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

coords_ID = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape
    # Extract Region of interest
    # roi = frame[int(height / 4): int(3 / 4 * height), int(width / 4):int(3 / 4 * width)]
    roi1 = frame[0:400, 0:400]
    # roi2 = frame[500:900, 100:800]

    # 1. Object Detection
    # применение маски к конкретному участку изображения
    mask = object_detector.apply(frame)
    # выделение пороговых значений - перевод всех цветов в черный-белый
    # без треша хуже идёт подсчет машин
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # выделение контуров в пороговых значениях
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 7000:
            # cv2.drawContours(roi1, [cnt], -1, (0, 255, 0), 2)
            # ограничение прямоугольника
            x, y, w, h = cv2.boundingRect(cnt)
            # types = classify(frame[y:y+h, x:x+w])
            # if(len(types) > 0):
            # добавление прграниц прямоугольника в detections
            detections.append([x, y, w, h])
            # print(types)

    # 2. Object Tracking
    # апдейт трекера массивом detections
    boxes_ids = tracker.update(detections)
    coords_ID.append(boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # cv2.imshow("roi1", roi1)
    cv2.imshow("Frame", frame)

    cv2.imshow("Mask", mask)
    for b in coords_ID:
        if len(b) == 0:
            coords_ID.remove(b)
    key = cv2.waitKey(30)
    if key == 27:
        break



# for b in coordsID:
print(coords_ID)
#     print("LEN: ")
#     print(len(b))
# max_id = coords_ID[0][0][4]
max_id = 0
for i in range(len(coords_ID) - 1):
    if coords_ID[i+1][0][4] > coords_ID[i][0][4]:
        max_id = coords_ID[i+1][0][4]
print(max_id)

print(coords_ID[1][0][4] > coords_ID[0][0][4])

print(tracker.coords_ID)
print("------------------------------")

diap1 = []
for i in range(100, 500, 1):
    diap1.append(i)

diap1_1_x = []
for i in range(600, 1000, 1):
    diap1_1_x.append(i)

diap1_1_y = []
for i in range(1100, 1700, 1):
    diap1_1_y.append(i)

maxCx = 0
maxCy = 0


temp1 = 0
# создаем массив из траектории конкретного объекта
for i in range(1, 5, 1):
    mass = []
    for ids in tracker.coords_ID:
        if ids[0] == i:
            mass.append(ids)
        else:
            continue
    print(str(i) + ')')
    print(mass)
    # print(mass[0][1][0])
    # print(mass[0][1][1])
    # if mass[0][1][0] in diap1 and mass[0][1][1] in diap1:
    #     if mass[len(mass)-1][1][0] in diap1_1_x and mass[len(mass)-1][1][1] in diap1_1_y:
    #         temp1 += 1
    print('first x: '+ str(mass[0][1][0]) + ' first y: ' + str(mass[0][1][1]))
    print('last x: '+ str(mass[len(mass) - 1][1][0]) + ' first y: ' + str(mass[len(mass) - 1][1][1]))







# for i in mass:
#     if mass[0][1][0]
# minCx = mass[0][1][0]
# minCy = mass[0][1][1]
# находим наименьшие и наибольшие координаты
# for i in range(len(mass)):
#     if mass[i][1][0] < minCx:
#         minCx = mass[i][1][0]
#         minCy = mass[i][1][1]
#     if mass[i][1][0] > maxCx:
#         maxCx = mass[i][1][0]
#         maxCy = mass[i][1][1]
#
# print(mass)
# print(minCx, minCy)
# print(maxCx, maxCy)
print("temp1 - " + str(temp1))
cap.release()
cv2.destroyAllWindows()

