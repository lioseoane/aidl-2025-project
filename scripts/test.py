import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from src.models.baseline_heatmap import baseline_heatmap
from torchvision import transforms
from src.utils.heatmaps import extract_keypoints_with_confidence
from src.data.dataloader import create_dataloaders
from src.data.load_workout_data import load_workout_data

# Initialize model and load checkpoint
model = baseline_heatmap(num_classes=20, num_keypoints=17, backbone='resnet50')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
checkpoint_path = 'checkpoints/model_epoch_5.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()


image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get original dimensions
h, w, _ = image.shape
target_w, target_h = [224, 224]

scale_w = target_w / float(w)
scale_h = target_h / float(h)

if scale_w == 1.0 and scale_h == 1.0:
    pass
else:
    scale = min(scale_w, scale_h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

    if  scale_w != scale_h:
        # Calculate padding
        pad_top = (target_h  - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w  - new_w - pad_left

        # Add padding
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
sample_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
sample_tensor = sample_tensor.unsqueeze(0).to(device)

# For visualization, we need to "unnormalize" the sample.
# The normalization used was: (x - mean) / std, so we reverse that.
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Move the sample to CPU and convert to numpy
sample_img = sample_tensor[0].cpu().detach().numpy()  # shape: [C, H, W]
sample_img = np.transpose(sample_img, (1, 2, 0))  # shape: [H, W, C]
# Unnormalize
sample_img = sample_img * std + mean
sample_img = np.clip(sample_img, 0, 1)

with torch.no_grad():
    output = model(sample_tensor)

bbox_pred = output[0]               # Predicted bounding boxes (for the sample)
heatmap_pred = output[1]            # Heatmap tensor [1, 17, H, W]
keypoints_pred = extract_keypoints_with_confidence(heatmap_pred)
workout_label_pred = output[2]

print("BBox Prediction:", bbox_pred)
print("Keypoints Prediction:", keypoints_pred)
print("Workout Label Prediction:", workout_label_pred)

# Sum the heatmap over the batch and keypoint channels to get a single [H, W] array.
# Since we passed one sample, axis 0 and 1 correspond to batch and keypoint channels.
summed_heatmap = np.sum(heatmap_pred.cpu().detach().numpy(), axis=(0, 1))

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the unnormalized (preprocessed) image
axs[0].imshow(sample_img)
axs[0].set_title("Preprocessed Image (Unnormalized)")
axs[0].axis("off")

# Display the summed keypoint heatmap with the 'jet' colormap
axs[1].imshow(summed_heatmap, cmap="jet")
axs[1].set_title("Summed Keypoint Heatmap")
axs[1].axis("off")

plt.tight_layout()
plt.show()
