import cv2

cap = cv2.VideoCapture("../videos/Road traffic video for object recognition.mp4")
subtrac = cv2.BackgroundSubtractor

points = []

cars = 0
while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    if not ret:
        break
    points.append((0, w - 150))
    points.append((int(h / 2) - 100, int(w / 2) - 50))
    points.append((int(h / 2) + 150, int(w / 2) - 50))
    points.append((h - 1, w - 1))
    points.append((0, w - 1))

    vecofp = [].append(points)
    filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.fillPoly(filter, vecofp, (255, 255, 255))
    copy = frame
    cv2.bitwise_and(frame, frame, copy, filter)
    sub = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(copy, (9, 9), cv2.BORDER_DEFAULT)
    # subtrac.apply(copy, sub)
    subtrac.apply(subtrac, copy)
    ker = cv2.getStructuringElement(cv2.fitEllipse(points), (5, 5), (-1, -1))

    print(4)
