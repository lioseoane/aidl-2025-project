import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import cv2
from src.data.load_workout_data import load_workout_data
from src.models.fast_rcnn import Fast_RCNN
from src.data.dataset import NewKeypointDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keypoints_array, images_array, head_boxes_array = load_mpii_data()
keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

num_classes = len(set(classes_array)) + 1 
num_keypoints = 17
model = Fast_RCNN(num_classes=num_classes, num_keypoints=num_keypoints)

model_path = "keypoint_model.pth"
model.load_state_dict(torch.load(model_path))

# Load the trained model
model.eval()
model.to(device)

# Load a single image from the dataset
dataset = NewKeypointDataset(images_array, bounding_boxes_array, keypoints_array, classes_array, resize_to=224)
image_idx = 1  # Change this index to test different images
image, target = dataset[image_idx]

# Prepare the image
image = image.to(device).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    predictions = model(image)

print(predictions)
# Process predictions
predicted_boxes = predictions[0]['boxes'].cpu().numpy()
predicted_labels = predictions[0]['labels'].cpu().numpy()
predicted_scores = predictions[0]['scores'].cpu().numpy()

# Filter predictions (e.g., keep those with high confidence)
threshold = 0.5
filtered_boxes = predicted_boxes[predicted_scores >= threshold]
filtered_labels = predicted_labels[predicted_scores >= threshold]

# Visualize the results
image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert back to NumPy
image_np = (image_np * 255).astype('uint8')  # Denormalize

for box, label in zip(filtered_boxes, filtered_labels):
    x1, y1, x2, y2 = box
    cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image_np, str(label), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Plot the image
plt.imshow(image_np)
plt.axis('off')
plt.show()