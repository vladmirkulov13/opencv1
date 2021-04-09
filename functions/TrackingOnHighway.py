from tracker import *
from Detection import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("../videos/perekrestok_Trim.mp4")

# Object detection from Stable camera
# маска на основе гаусса
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

coords_ID = []


def removeBigDiff(mass):
    if len(mass) < 30:
        return mass
    else:
        i = 0
        while i < len(mass) - 1:
            if mass[i] != 0:
                if abs(mass[i + 1][1][0] - mass[i][1][0]) > 100 and abs(mass[i + 1][1][1] - mass[i][1][1]) > 100:
                    del mass[i + 1]

                else:
                    i += 1
        return mass


def changeArray(masses):
    i = 0
    while i < len(masses):
        x = masses[i][-1][1][0]
        y = masses[i][-1][1][1]
        j = i + 1
        while j < len(masses):
            x_n = masses[j][0][1][0]
            y_n = masses[j][0][1][1]
            if abs(x_n - x) < 20 and abs(y_n - y) < 20:
                for q in masses[j]:
                    masses[i].append(q)
                del masses[j]
            else :
                j += 1

        i += 1



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
        if area > 500:
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
    if coords_ID[i + 1][0][4] > coords_ID[i][0][4]:
        max_id = coords_ID[i + 1][0][4]
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
masses = []
for i in range(0, 100, 1):
    mass = []
    for ids in tracker.coords_ID:
        if ids[0] == i:
            mass.append(ids)
        else:
            continue

    if len(mass) == 0:
        continue
    # print(str(i) + ')')
    # print(mass)
    mass = removeBigDiff(mass)
    masses.append(mass)
    # print(mass)

    # if mass[0][1][0] in diap1 and mass[0][1][1] in diap1:
    #     if mass[len(mass)-1][1][0] in diap1_1_x and mass[len(mass)-1][1][1] in diap1_1_y:
    #         temp1 += 1
    # print('(' + str(mass[0][1][0]) + ', ' + str(mass[0][1][1]) + ')')
    # print('(' + str(mass[-1][1][0]) + ', ' + str(mass[-1][1][1]) + ')')
    # if len(mass) > 30:
    #     massProof.append(mass)
    # else:
    #     massLess.append(mass)
# changeArrays(massProof, massLess)
# print(mass)
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
# i = 0
# while i < len(masses):
#     x_0 = masses[i][0][1][0]
#     y_0 = masses[i][0][1][1]
#     x_n = masses[i][-1][1][0]
#     y_n = masses[i][-1][1][1]
#     dif_x = x_n - x_0
#     dif_y = y_n - y_0
#     if dif_x < 80 and dif_y < 80:
#         del masses[i]
#     else:
#         i += 1
changeArray(masses)

print("temp1 - " + str(temp1))
cap.release()
cv2.destroyAllWindows()
