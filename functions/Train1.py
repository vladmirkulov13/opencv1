import cv2
import Detection
import SiftTracker

cap = cv2.VideoCapture("../videos/Road traffic video for object recognition.mp4")
_, first_frame = cap.read()
# cv2.imwrite("first_frame.jpg", frame)
_, cars = Detection.classify(first_frame)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    for i in range(len(cars)):
        img = first_frame[cars[i].coords[1]:cars[i].coords[3], cars[i].coords[0]: cars[i].coords[2]]
        SiftTracker.sift_track(frame, img)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(30) == 27:
        break
