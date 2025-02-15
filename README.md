﻿# YOLOv8-AI-trained-model-of-Nudes

# YOLO Object Detection with OpenCV and cvzone

This Python script demonstrates real-time object detection using the YOLO (You Only Look Once) model, OpenCV, and cvzone library. The application accesses a video feed, applies the YOLO model for object detection, and blurs the detected objects in the video stream.

# Dependencies

Ensure you have the necessary dependencies installed before running the script:

* Ultralytics YOLO: Follow the instructions to install YOLO and download the pre-trained weights (best.pt).
* OpenCV: pip install opencv-python
* cvzone: pip install cvzone

# How to Use

1. Clone the repository to your local machine:
* git clone https://github.com/your-username/yolo-object-detection.git

2. Navigate to the project directory:
* cd yolo-object-detection

3. Copy the pre-trained YOLO weights (best.pt) to the project directory.

4. Run the script:
*  python nude-detector.py

5. Press the 'q' key to exit the application.

# Configuration

* The script uses the YOLO model with pre-trained weights (best.pt). Ensure the weights file is available in the project directory.
*  Class names related to detections in the YOLO model are defined in the classnames list. Modify this list according to your specific use case.

#  Features
* Objects detected by the YOLO model with confidence greater than 50% are highlighted with bounding boxes.
* Detected objects are blurred in real-time, providing privacy or obscuring sensitive information.

#  Acknowledgments

* This project is based on the YOLO model by Ultralytics. Visit Ultralytics YOLO for more information.
* The cvzone library is used for drawing rectangles and text labels on the frame.
