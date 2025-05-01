import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 Nano (for speed) or Small (better accuracy)
model = YOLO("yolov8s.pt")  # Change to "yolov8s.pt" for Small model

# Open video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

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

    # ========== License Plate Detection Placeholder ==========
    # TODO: Load and run a separate License Plate model here
    # Example: lp_results = lp_model(frame)
    # Parse lp_results and draw license plate boxes

    # ========== Car Logo Detection Placeholder ==========
    # TODO: Load and run a separate Car Logo model here
    # Example: logo_results = logo_model(frame)
    # Parse logo_results and draw logo boxes

    # Display the output
    cv2.imshow("YOLOv8 Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
