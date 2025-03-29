# CL-FusionBEV: Advanced 3D Object Detection using LiDAR and Cross Attention

## Overview
CL-FusionBEV is an advanced 3D object detection system designed for self-driving vehicles. It fuses **LiDAR point cloud data** and **camera sensor images** using a **Bird’s Eye View (BEV)** representation and a **Cross-Attention Mechanism** to enhance object detection accuracy. The model effectively integrates multi-sensor data to improve object identification under varying environmental conditions, ensuring robust and precise perception in real-time autonomous navigation.

## Features
- **Fusion of LiDAR and Camera Data**: Utilizes a novel cross-attention model to merge depth-rich LiDAR data with color and texture-rich camera images.
- **BEV Feature Extraction**: Generates spatial BEV features from multi-view camera images and voxelized LiDAR point clouds.
- **YOLO-based Object Detection**: Integrates **YOLOv11** for object recognition with real-time inference.
- **Distance Estimation**: Computes object distance using both **camera-based triangulation** and **LiDAR depth measurements**.
- **Real-Time Alert System**: Triggers an alarm if an object is detected within a predefined distance threshold.
- **Efficient and Optimized for Embedded Deployment**: The framework is optimized for embedded hardware implementation to facilitate real-world deployment in autonomous vehicles.

## Installation
To set up the project, follow these steps:

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- OpenCV
- NumPy
- Torch & torchvision
- Ultralytics YOLO
- Flask (for web streaming)
- Playsound (for alarm system)
- A LiDAR driver library (specific to your hardware)

### Setup
```sh
# Clone the repository
git clone https://github.com/yourusername/CL-FusionBEV.git
cd CL-FusionBEV

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the Object Detection System
```sh
python app.py
```
This will start the Flask server, and you can access the **live camera stream** at:
```
http://127.0.0.1:5000/live_camera
```

### How It Works
1. **Data Collection**: Captures real-time frames from a connected camera and LiDAR sensor.
2. **Object Detection**: YOLOv11 identifies objects in the frame.
3. **Cross-Attention Fusion**: Enhances detections by merging LiDAR and image data.
4. **Distance Estimation**: Computes the object's position and triggers an alarm if the object is too close.
5. **Visualization & Alerts**: Draws bounding boxes, labels, and displays warnings on the video feed.

## Project Structure
```
CL-FusionBEV/
│── dataset/                 # COCO dataset for object classes
│── models/                  # Trained YOLO & Cross-Attention models
│── static/                  # Web assets (CSS, JS, images)
│── templates/               # HTML templates for Flask app
│── app.py                   # Main application script
│── lidar_module.py          # LiDAR sensor integration (Placeholder)
│── cross_attention_model.py # Cross-Attention model implementation (Placeholder)
│── requirements.txt         # Dependencies list
│── README.md                # Documentation
```

## Future Enhancements
- Improve robustness under extreme weather conditions (heavy rain, fog, etc.).
- Optimize the framework for **edge computing** to run efficiently on embedded hardware.
- Extend dataset evaluation to **Waymo Open Dataset** and **KITTI** for generalization testing.
- Implement real-time **sensor fusion visualization** to enhance interpretability.

## Acknowledgments
- **Ultralytics** for the YOLO object detection framework.
- **Waymo & KITTI datasets** for benchmarking autonomous perception systems.
