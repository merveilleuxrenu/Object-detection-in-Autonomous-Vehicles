import cv2
import numpy as np
import time
from ultralytics import YOLO
from playsound import playsound
from threading import Thread



# Load the YOLOv11 model
model = YOLO("yolo11n.pt")  # Make sure you have the trained YOLOv11 model file

# Constants
focal_length = 150  # Camera focal length
known_width = 10.0  # Known width of the object in cm
distance_threshold = 0.10  # Distance threshold for alarm (meters)

classNames = []
classFile = "dataset/coco.txt"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


alarm_on = False
alarm_sound = "data/alarm.mp3"

def start_alarm(sound):
    """Play the alarm sound"""
    playsound('alarm.mp3')

def get_distance(known_width, focal_length, pixel_width):
    """Calculate object distance from the camera."""
    return (known_width * focal_length) / pixel_width

def detect_objects(img):
    """Perform object detection and calculate distance."""
    results = model(img)
    object_info = []

    for r in results:
        boxes = r.boxes  # Get bounding boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            pixel_width = x2 - x1  # Object width in pixels
            class_id = int(box.cls[0])  # Object class ID
            confidence = float(box.conf[0])  # Confidence score

            # Calculate distance
            distance = get_distance(known_width, focal_length, pixel_width) * 0.0254  # Convert to meters

            # Display bounding box and information
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Object: {classNames[class_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Conf: {confidence:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Dist: {distance:.2f} m", (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Trigger an alarm if the object is too close
            if distance < distance_threshold:
                cv2.putText(img, "STOP! Object too close", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                #if not alarm_on:
                alarm_on = True
                print('alarm')
                t = Thread(target=start_alarm, args=(alarm_sound,))
                t.daemon = True
                t.start()

            object_info.append((class_id, confidence, distance))
            alarm_on = False

    return img, object_info

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Process frame
    result_frame, detected_objects = detect_objects(frame)

    # Show the processed frame
    cv2.imshow("YOLOv11 Distance-Based Detection", result_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
