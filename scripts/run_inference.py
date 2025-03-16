import os
import sys
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from src.models.heatmap_fpn_v3 import heatmap_fpn
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps

def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    img_tensor: Tensor (C x H x W) normalized image
    returns: NumPy array (H x W x C) in [0, 1]
    """
    img = img_tensor.clone().cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


# Initialize model and load checkpoint
model = heatmap_fpn(num_classes=20, num_keypoints=17, backbone='resnet50') # Model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
checkpoint_path = 'checkpoints/model_epoch_50.pth' # Model checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Load and preprocess image
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
target_w, target_h = [352, 352]

scale = min(target_w / float(w), target_h / float(h))
new_w, new_h = int(w * scale), int(h * scale)

preprocessed_image = cv2.resize(image, (new_w, new_h))

pad_top = (target_h - new_h) // 2
pad_bottom = target_h - new_h - pad_top
pad_left = (target_w - new_w) // 2
pad_right = target_w - new_w - pad_left

preprocessed_image = cv2.copyMakeBorder(
    preprocessed_image, pad_top, pad_bottom, pad_left, pad_right,
    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
)

preprocessed_image_tensor = torch.tensor(preprocessed_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
preprocessed_image_tensor = (preprocessed_image_tensor - mean) / std
preprocessed_image_tensor = preprocessed_image_tensor.unsqueeze(0).to(device)

# Unnormalize for visualization
image_preprocessed = preprocessed_image_tensor[0].cpu().detach().numpy()
image_preprocessed = np.transpose(image_preprocessed, (1, 2, 0))

with torch.no_grad():
    output = model(preprocessed_image_tensor)

bbox_heatmap_pred = output[0]               
bbox_pred = extract_bbox_from_heatmaps(output[0]) 
heatmap_pred = output[1]            
keypoints_pred = extract_keypoints_with_confidence(heatmap_pred)
workout_label_pred = output[2] 

bbox_heatmap_final = bbox_heatmap_pred.squeeze(0).cpu().detach().numpy()

image_preprocessed = denormalize_image(preprocessed_image_tensor[0])

# Sum the heatmap over the keypoint channels to get a single [H, W] array
summed_heatmap = np.sum(heatmap_pred.cpu().detach().numpy(), axis=(0, 1))

for bbox in bbox_pred:
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image_preprocessed, (int(x_min * target_w), int(y_min * target_h)), (int(x_max * target_w), int(y_max * target_h)), (0, 255, 0), 1)

# Draw keypoints on the original image1
filtered_keypoints = []
for i, point in enumerate(keypoints_pred[0]):
    x, y, confidence = point
    if confidence >= 0.5:  # Only draw visible keypoints
        cv2.circle(image_preprocessed, (int(x * target_w), int(y * target_h)), 1, (1, 0, 0), -1)  # Red dots for keypoints
        filtered_keypoints.append((i, x, y))  # Store valid keypoints for skeleton drawing

SKELETON = [
    (0, 1), (1, 2), (2, 0),     # Nose -> Left Eye -> Right Eye -> Nose,
    (1, 3),                     # Left Eye -> Left Ear
    (2, 4),                     # Right Eye -> Right Ear
    (5, 6),                     # Left Shoulder -> Right Shoulder
    (5, 7), (7, 9),             # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),            # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12),           # Left Shoulder -> Left Hip, Right Shoulder -> Right Hip
    (11, 13), (13, 15),         # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16)          # Right Hip -> Right Knee -> Right Ankle
]

# Draw the skeleton using only valid keypoints
for pair in SKELETON:
    i, j = pair
    valid_points = {kp[0]: (kp[1], kp[2]) for kp in filtered_keypoints}
    if i in valid_points and j in valid_points:
        x1, y1 = valid_points[i]
        x2, y2 = valid_points[j]
        cv2.line(image_preprocessed, (int(x1 * target_w), int(y1 * target_h)), 
                     (int(x2 * target_w), int(y2 * target_h)), (0, 0, 255), 1)  # Blue lines


with open('resnet50_2025-03-15_19-00-46.json', 'r') as f:
    idx_to_class_name = json.load(f)

# Display workout class and probability
probabilities = torch.softmax(workout_label_pred[0], dim=0) 
predicted_class_idx = torch.argmax(probabilities).item()
predicted_class_name = idx_to_class_name[str(predicted_class_idx)]  # Map index to class name
print(predicted_class_name, " - prob: ",  probabilities[predicted_class_idx].item())

# Create a figure with two subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Display the preprocessed image with keypoints
axs[0].imshow(image_preprocessed)
axs[0].set_title("Preprocessed Image with Keypoints")
axs[0].axis("off")

# Display the predicted heatmap
axs[1].imshow(summed_heatmap, cmap="jet")
axs[1].set_title("Keypoint Heatmap Predicted")
axs[1].axis("off")

# Display the predicted heatmap
axs[2].imshow(bbox_heatmap_final, cmap="jet")
axs[2].set_title("BBox Heatmap Predicted")
axs[2].axis("off")

plt.tight_layout()
plt.show()