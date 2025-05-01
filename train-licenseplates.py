from ultralytics import YOLO
import os
import glob

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # "s" is more accurate; use "n" for speed

# Create a copy of the fixed data.yaml in the dataset folder
data_yaml_path = "/Users/nick/Projects/Python/Deep and Machine Learning/Using YOLOv8 for Multi-Attribute Detection in Vehicles/License-Plate-Recognition-6/data.yaml"

# Write the fixed content to the data.yaml file
fixed_data_yaml = """# Classes
names:
  - License_Plate
nc: 1

# Paths to training/validation/test sets (without relative paths)
train: train/images
val: valid/images
test: test/images

# Dataset information
roboflow:
  license: CC BY 4.0
  project: license-plate-recognition-rxg4e
  url: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/6
  version: 6
  workspace: roboflow-universe-projects
"""

# Write the fixed YAML content to the file
with open(data_yaml_path, "w") as f:
    f.write(fixed_data_yaml)

print(f"Updated data.yaml file at: {data_yaml_path}")

# Train with path to data.yaml
model.train(
    data=data_yaml_path,
    epochs=50, 
    imgsz=640, 
    batch=16, 
    device="cuda",
    patience=10,
    save=True,
    lr0=0.01,
    lrf=0.01,
    augment=True,
    name="license_plate_detector"
)

# Validate the model
metrics = model.val()
print("Validation metrics:", metrics)

# Now test on actual test images
print("\n=== Testing on sample images ===\n")

# Path to test images folder
test_folder = "/Users/nick/Projects/Python/Deep and Machine Learning/Using YOLOv8 for Multi-Attribute Detection in Vehicles/License-Plate-Recognition-8/test/images"
print(f"Looking for images in: {test_folder}")

# Find all test images
image_extensions = ['*.jpg', '*.jpeg', '*.png']
test_images = []

for extension in image_extensions:
    found_images = glob.glob(os.path.join(test_folder, extension))
    test_images.extend(found_images)
    print(f"Found {len(found_images)} images with extension {extension}")

if not test_images:
    print(f"No image files found in {test_folder}")
    exit()

print(f"Found {len(test_images)} total test images")

# Test the model on some sample images
num_samples = min(5, len(test_images))  # Test on up to 5 images
print(f"\nTesting on {num_samples} sample images:")

for i, test_image in enumerate(test_images[:num_samples]):
    print(f"\nProcessing image {i+1}: {os.path.basename(test_image)}")
    
    # Run inference
    results = model(test_image, save=True, conf=0.25)
    
    # Print detection results
    for r in results:
        boxes = r.boxes
        print(f"Found {len(boxes)} license plates")
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            print(f"  License plate detected at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")

print("\nResults are saved in the 'runs/detect/predict' folder")