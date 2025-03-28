import cv2
import numpy as np
import time
from ultralytics import YOLO
from playsound import playsound
from threading import Thread

# Load YOLO model
model = YOLO("yolo11n.pt")  

# Distance parameters
focal_length = 150  
known_width = 10.0  
distance_threshold = 0.10  # Distance at which alarm triggers
max_detection_distance = 2.0  # Objects beyond this distance won't be detected

# Load class names
classNames = []
classFile = "dataset/coco.txt"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

alarm_on = False
alarm_sound = "data/alarm.mp3"

def start_alarm(sound):
    playsound(sound)

def get_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

def detect_objects(img):
    results = model(img)
    object_info = []
    
    # Create an empty feature map (same size as the frame)
    feature_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for r in results:
        boxes = r.boxes  

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            pixel_width = x2 - x1  
            class_id = int(box.cls[0])  
            confidence = float(box.conf[0])  

            # Calculate distance in meters
            distance = get_distance(known_width, focal_length, pixel_width) * 0.0254  

            # **Filter out objects beyond max_detection_distance**
            if distance > max_detection_distance:
                continue  # Skip this object

            # Normalize the distance (higher distance = lower intensity)
            intensity = max(0, min(255, int(255 * (1 - (distance / max_detection_distance)))))  
            feature_map[y1:y2, x1:x2] = intensity  # Assign intensity to bounding box area

            # Draw detections
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Object: {classNames[class_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Conf: {confidence:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Dist: {distance:.2f} m", (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if distance < distance_threshold:
                cv2.putText(img, "STOP! Object too close", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                #if not alarm_on:
                alarm_on = True
                t = Thread(target=start_alarm, args=(alarm_sound,))
                t.daemon = True
                t.start()

            object_info.append((class_id, confidence, distance))

    # Convert feature map to a heatmap
    heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)

    return img, heatmap, object_info

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    result_frame, heatmap, detected_objects = detect_objects(frame)

    # Display both object detection and heatmap
    combined_frame = cv2.addWeighted(result_frame, 0.6, heatmap, 0.4, 0)

    cv2.imshow("YOLOv11 Distance-Based Detection", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
