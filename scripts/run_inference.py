import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import cv2
from src.data.load_workout_data import load_workout_data
from src.models.heatmap_lateral import heatmap_lateral
from src.data.dataset import WorkoutDataset
from src.utils.heatmaps import extract_keypoints_with_confidence
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keypoints_array, images_array, head_boxes_array = load_mpii_data()
keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

size_x = 224
size_y = 224
num_classes = 20
num_keypoints = 17
model = heatmap_lateral(num_classes=num_classes, num_keypoints=num_keypoints, backbone='resnet50')

model_path = "checkpoints/model_epoch_55.pth"
model.load_state_dict(torch.load(model_path))

# Load the trained model
model.eval()
model.to(device)

# Load a single image from the dataset
dataset = WorkoutDataset(images_array, bounding_boxes_array, keypoints_array, classes_array, resize_to=[size_x, size_y], sigma=2)

image_idx = 1  # Change this index to test different images
image, target = dataset[image_idx]

# Prepare the image
image = image.to(device).unsqueeze(0)  # Add batch dimension

# Skeleton structure for YOLO 17 keypoints (pairs of keypoints to connect)
SKELETON = [
    (0, 1), (1, 2), (2, 3), (2, 4),  # Nose -> Left Eye -> Right Eye, Right Eye -> Right Ear, Left Eye -> Left Ear
    (5, 6),                          # Left Shoulder -> Right Shoulder
    (5, 7), (7, 9),                  # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),                 # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12),                # Left Shoulder -> Left Hip, Right Shoulder -> Right Hip
    (11, 13), (13, 15),              # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16)               # Right Hip -> Right Knee -> Right Ankle
]


# Make prediction
with torch.no_grad():
    predictions = model(image)


# Process predictions
bbox_pred = predictions[0] 
heatmap_pred = predictions[1]
keypoints_pred = extract_keypoints_with_confidence(heatmap_pred)  
workout_label_pred = predictions[2]  

# Sum the heatmap over the keypoint channels to get a single [H, W] array
summed_heatmap = np.sum(heatmap_pred.cpu().detach().numpy(), axis=(0, 1))

# Draw the bounding boxes (bbox_pred should be in (x_min, y_min, x_max, y_max) format)
for bbox in bbox_pred:
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image[0], (int(x_min * size_x), int(y_min * size_y)), (int(x_max * size_x), int(y_max * size_y)), (0, 255, 0), 2)

# Draw the keypoints (keypoints_pred should be a tensor with shape [num_keypoints, 2] for x, y)
keypoints_pred = keypoints_pred[0]
filtered_keypoints = []

for i, point in enumerate(keypoints_pred):
    x, y, confidence = point

    if confidence >= 0.5 and x_min <= x <= x_max and y_min <= y <= y_max:
        cv2.circle(image[0], (int(x * size_x), int(y * size_y)), 5, (0, 0, 255), -1)  # Red dots for keypoints

        cv2.putText(image[0], str(i), (int(x * size_x) + 10, int(y * size_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
            
        filtered_keypoints.append((i, x, y))  # Store valid keypoints for skeleton drawing

# Draw the skeleton using only valid keypoints
for pair in SKELETON:
    i, j = pair
    valid_points = {kp[0]: (kp[1], kp[2]) for kp in filtered_keypoints}
    if i in valid_points and j in valid_points:
        x1, y1 = valid_points[i]
        x2, y2 = valid_points[j]
        cv2.line(image[0], (int(x1 * size_x), int(y1 * size_y)), 
                     (int(x2 * size_x), int(y2 * size_y)), (255, 0, 0), 2)  # Blue lines
        
# Draw keypoints on the heatmap
heatmap_with_keypoints = np.copy(summed_heatmap)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the preprocessed image with keypoints
axs[0].imshow(image[0])
axs[0].set_title("Preprocessed Image with Keypoints")
axs[0].axis("off")

# Display the summed heatmap with keypoints
axs[1].imshow(heatmap_with_keypoints, cmap="jet")
axs[1].set_title("Summed Keypoint Heatmap with Keypoints")
axs[1].axis("off")

plt.tight_layout()
plt.show()
