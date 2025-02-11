import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Replace with the path where your dataset is extracted
data_dir = 'data/'

# Collect all image files from subdirectories
image_files = []
for state_folder in os.listdir(data_dir):
    state_path = os.path.join(data_dir, state_folder)
    if os.path.isdir(state_path):
        for file in os.listdir(state_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(state_path, file))

# Display the first 10 images
plt.figure(figsize=(15, 8))

for i in range(min(10, len(image_files))):
    img_path = image_files[i]
    img = mpimg.imread(img_path)

    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(os.path.basename(img_path))
    plt.axis('off')

plt.tight_layout()
plt.show()
