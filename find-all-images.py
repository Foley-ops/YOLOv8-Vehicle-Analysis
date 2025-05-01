import os
import glob

# Base path to the state license plates dataset
base_path = "/Users/nick/Projects/Python/Deep and Machine Learning/Using YOLOv8 for Multi-Attribute Detection in Vehicles"
states_path = os.path.join(base_path, "licenseplate-states")
roboflow_path = os.path.join(base_path, "License-Plate-Recognition-8")

# Function to find all directories containing images
def find_image_dirs(root_dir):
    image_dirs = []
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if directory contains images
        has_images = False
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            if glob.glob(os.path.join(dirpath, ext)):
                has_images = True
                break
        
        if has_images:
            image_dirs.append(dirpath)
    
    return image_dirs

# Find all directories with images
image_dirs = find_image_dirs(states_path)
print(f"Found {len(image_dirs)} directories with images")

# Generate the YAML file
yaml_content = f"""# Auto-generated combined license plate dataset YAML

# Classes
names:
  - License_Plate
nc: 1

# Training paths - combining both datasets
train: [
  "{os.path.join(roboflow_path, 'train', 'images')}",
"""

# Add all image directories to the YAML
for image_dir in image_dirs:
    yaml_content += f'  "{image_dir}",\n'

# Remove the trailing comma and close the list
yaml_content = yaml_content.rstrip(',\n') + "\n]\n\n"

# Add validation and test paths
yaml_content += f"""# Validation path
val: "{os.path.join(roboflow_path, 'valid', 'images')}"

# Test path
test: "{os.path.join(roboflow_path, 'test', 'images')}"
"""

# Write to file
yaml_path = os.path.join(base_path, "combined_license_plates.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Generated YAML file at: {yaml_path}")
print("\nFirst few image directories:")
for dir in image_dirs[:5]:
    print(f"- {dir}")