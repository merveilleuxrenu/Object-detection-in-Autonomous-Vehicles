from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import threading
import time
import os
import image_dehazer


app = Flask(__name__)

model = YOLO("yolo11n.pt")  

# Initialize the webcam

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame)
        
        # Draw results on the frame
        for r in results:
            frame = r.plot()  # Draw detected objects on the frame

        # Convert frame to JPEG to send to frontend
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame = buffer.tobytes()

        # Yield the frame for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_camera')
def live_camera():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_upload', methods=['POST'])
def image_upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
       # Save the uploaded image in the 'static' folder
    filename = os.path.join("static", "uploads", file.filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file.save(filename)

    # Load and process the uploaded image
    image = cv2.imread(filename)
    image = cv2.resize(image, (640, 480))  # Resize to 640x480
    HazeCorrectedImg, haze_map = image_dehazer.remove_haze(image, showHazeTransmissionMap=False)

    results = model(HazeCorrectedImg)
    
    # Save the result image with detections in the 'static' folder
    output_filename = os.path.join("static", "uploads", "output_" + file.filename)
    for r in results:
        image_with_boxes = r.plot()  # Draw boxes on detected objects
    cv2.imwrite(output_filename, image_with_boxes)
    
    # Return the image filename for frontend rendering
    return render_template('result.html', image_filename="uploads/" + "output_" + file.filename)

if __name__ == "__main__":
    app.run(debug=False)
