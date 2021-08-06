import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        # словарь номер машины : координаты её центра
        self.center_points = {}
        self.coords_ID = []
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        # self.center_points_last = {}


# передаётся массив координат ограничевающего прямоугольника
    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            # вычисление середи прямоугольника по х и по у
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # self.coords_ID.append([self.id_count, (cx, cy)])

            # Find out if that object was detected already
            same_object_detected = False
            # идем по словарю центров прямоугольников
            for id, pt in self.center_points.items():
                # сравнивается расстояние от центра координат до:
                # 1. вычисленных координат центра прямоугольника
                # 2. координат центров прошлого прямоугольника
                dist = math.hypot(cx - pt[0], cy - pt[1])
                # если сравнение точек дало < 100
                # значит объект тот же самый
                if dist < 25:
                    # перезаписываем в словаре значение для данного прямоугольника
                    self.center_points[id] = (cx, cy)
                    # temp = [id, (cx, cy)]
                    # self.coords_ID.append(temp)
                    # вывод номера объекта + координаты
                    # print(self.center_points)
                    # self.center_points_last[self.id_count].append(self.center_points.items())
                    # self.center_points_last.update(self.center_points)
                    # добавляем координаты и номер прямоугольника в objects_bbs_ids
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    # выход из цикла
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)

                objects_bbs_ids.append([x, y, w, h, self.id_count])
                # cx = (x + x + w) // 2
                # cy = (y + y + h) // 2
                # self.coords_ID.append([self.id_count, (cx, cy)])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        # if len(self.center_points) != 0:
        #     self.coords_ID.append(self.center_points.keys())
        #     self.coords_ID.append(self.center_points.items())
        return objects_bbs_ids



