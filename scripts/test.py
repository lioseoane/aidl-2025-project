import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from src.models.heatmap_lateral import heatmap_lateral
from torchvision import transforms
from src.utils.heatmaps import extract_keypoints_with_confidence

# Initialize model and load checkpoint
model = heatmap_lateral(num_classes=20, num_keypoints=17, backbone='resnet50')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
checkpoint_path = 'checkpoints/model_epoch_55.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Load and preprocess image
image = cv2.imread('test5q.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
target_w, target_h = [224, 224]

scale = min(target_w / float(w), target_h / float(h))
new_w, new_h = int(w * scale), int(h * scale)
image = cv2.resize(image, (new_w, new_h))

pad_top = (target_h - new_h) // 2
pad_bottom = target_h - new_h - pad_top
pad_left = (target_w - new_w) // 2
pad_right = target_w - new_w - pad_left

image = cv2.copyMakeBorder(
    image, pad_top, pad_bottom, pad_left, pad_right,
    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
)

sample_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
sample_tensor = sample_tensor.unsqueeze(0).to(device)

# Unnormalize for visualization
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
sample_img = sample_tensor[0].cpu().detach().numpy()
sample_img = np.transpose(sample_img, (1, 2, 0))
#sample_img = np.clip(sample_img * std + mean, 0, 1)

with torch.no_grad():
    output = model(sample_tensor)

bbox_pred = output[0]               # Predicted bounding boxes
heatmap_pred = output[1]            # Heatmap tensor [1, 17, H, W]
keypoints_pred = extract_keypoints_with_confidence(heatmap_pred)
print(keypoints_pred)

# Sum the heatmap over the keypoint channels to get a single [H, W] array
summed_heatmap = np.sum(heatmap_pred.cpu().detach().numpy(), axis=(0, 1))

# Draw keypoints on the original image
filtered_keypoints = []
for i, point in enumerate(keypoints_pred[0]):
    x, y, confidence = point
    if confidence >= 0.5:  # Only draw visible keypoints
        cv2.circle(sample_img, (int(x * target_w), int(y * target_h)), 5, (1, 0, 0), -1)  # Red dots for keypoints
        filtered_keypoints.append((i, x, y))  # Store valid keypoints for skeleton drawing

SKELETON = [
    (0, 1), (1, 2), (2, 3), (2, 4),  # Nose -> Left Eye -> Right Eye, Right Eye -> Right Ear, Left Eye -> Left Ear
    (5, 6),                          # Left Shoulder -> Right Shoulder
    (5, 7), (7, 9),                  # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),                 # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12),                # Left Shoulder -> Left Hip, Right Shoulder -> Right Hip
    (11, 13), (13, 15),              # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16)               # Right Hip -> Right Knee -> Right Ankle
]

# Draw the skeleton using only valid keypoints
for pair in SKELETON:
    i, j = pair
    valid_points = {kp[0]: (kp[1], kp[2]) for kp in filtered_keypoints}
    if i in valid_points and j in valid_points:
        x1, y1 = valid_points[i]
        x2, y2 = valid_points[j]
        cv2.line(image[0], (int(x1 * target_w), int(y1 * target_h)), 
                     (int(x2 * target_w), int(y2 * target_h)), (255, 0, 0), 2)  # Blue lines

# Draw keypoints on the heatmap
heatmap_with_keypoints = np.copy(summed_heatmap)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the preprocessed image with keypoints
axs[0].imshow(sample_img)
axs[0].set_title("Preprocessed Image with Keypoints")
axs[0].axis("off")

# Display the summed heatmap with keypoints
axs[1].imshow(heatmap_with_keypoints, cmap="jet")
axs[1].set_title("Summed Keypoint Heatmap with Keypoints")
axs[1].axis("off")

plt.tight_layout()
plt.show()
