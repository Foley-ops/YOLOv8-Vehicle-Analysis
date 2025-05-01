import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO

# Check for MPS (Metal Performance Shaders) on Mac
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 models and move to appropriate device
model = YOLO("yolov8s.pt").to(device)  # Main vehicle detection model
lp_model = YOLO("yolov8s-licenseplate.pt").to(device)  # License plate model

# Define vehicle classes to detect (COCO dataset class indices)
# 2:car, 5:bus, 7:truck, 3:motorcycle
VEHICLE_CLASSES = [2, 5, 7, 3]  
VEHICLE_NAMES = ['car', 'bus', 'truck', 'motorcycle']

# Video file path (change this to your video file)
video_path = "1001/0a3f01efd937dab6adc569614e635b68.mp4"

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file does not exist at {video_path}")
    exit()

# Open video file instead of webcam
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {video_path}")
print(f"Dimensions: {frame_width}x{frame_height}, FPS: {fps}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached")
        break
    
    # Run YOLOv8 inference on the frame with class filtering
    results = model(frame, classes=VEHICLE_CLASSES)  # Only detect vehicles

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ========== License Plate Detection ==========
    # Process only top 5 highest confidence vehicles for efficiency
    vehicle_boxes = []
    
    for result in results:
        for box in result.boxes:
            vehicle_boxes.append((box, box.conf[0]))
    
    # Sort by confidence and take top 5
    vehicle_boxes.sort(key=lambda x: x[1], reverse=True)
    vehicle_boxes = vehicle_boxes[:5]
    
    for box_info in vehicle_boxes:
        box = box_info[0]
        # Get vehicle box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Extract vehicle region
        vehicle_img = frame[y1:y2, x1:x2]
        
        # Only process if we have a valid vehicle region
        if vehicle_img.size > 0:
            # Detect license plates in the vehicle region
            lp_results = lp_model(vehicle_img)
            
            # Process license plate detections
            for lp_result in lp_results:
                for lp_box in lp_result.boxes:
                    # Get license plate coordinates relative to the vehicle crop
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box.xyxy[0])
                    lp_conf = float(lp_box.conf[0])
                    
                    # Convert coordinates to be relative to the full frame
                    lp_x1, lp_y1 = lp_x1 + x1, lp_y1 + y1
                    lp_x2, lp_y2 = lp_x2 + x1, lp_y2 + y1
                    
                    # Draw license plate bounding box
                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"License: {lp_conf:.2f}", (lp_x1, lp_y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the output
    cv2.imshow("YOLOv8 Vehicle Detection", frame)
    
    # Calculate appropriate delay based on video's FPS
    delay = int(1000 / fps * 4) if fps > 0 else 8  # Convert FPS to millisecond delay
    
    # Press 'q' to exit
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()