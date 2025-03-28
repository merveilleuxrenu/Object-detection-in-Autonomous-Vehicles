import cv2
import numpy as np
import time
from ultralytics import YOLO
from playsound import playsound
from threading import Thread



model = YOLO("yolo11n.pt")  

focal_length = 150  
known_width = 10.0  
distance_threshold = 0.10  

classNames = []
classFile = "dataset/coco.txt"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


alarm_on = False
alarm_sound = "data/alarm.mp3"

def start_alarm(sound):
    playsound('alarm.mp3')

def get_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

def detect_objects(img):
    results = model(img)
    object_info = []

    for r in results:
        boxes = r.boxes  

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            pixel_width = x2 - x1  
            class_id = int(box.cls[0])  
            confidence = float(box.conf[0])  

            distance = get_distance(known_width, focal_length, pixel_width) * 0.0254  

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Object: {classNames[class_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Conf: {confidence:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Dist: {distance:.2f} m", (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if distance < distance_threshold:
                cv2.putText(img, "STOP! Object too close", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                alarm_on = True
                print('alarm')
                t = Thread(target=start_alarm, args=(alarm_sound,))
                t.daemon = True
                t.start()

            object_info.append((class_id, confidence, distance))
            alarm_on = False

    return img, object_info

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    result_frame, detected_objects = detect_objects(frame)

    cv2.imshow("YOLOv11 Distance-Based Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
