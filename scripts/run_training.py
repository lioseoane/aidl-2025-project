import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataloader import create_dataloaders
from src.data.load_workout_data import load_workout_data
from src.models.fast_rcnn import Fast_RCNN
from src.training.train import train_model
import torch
import torch.optim as optim

# Load data (images and keypoints)
keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

# Create dataloaders
train_loader, val_loader = create_dataloaders(images_array, bounding_boxes_array, keypoints_array, classes_array, batch_size=64, resize_to=128)

# Initialize model, optimizer, and loss function
num_classes = len(set(classes_array)) 
num_keypoints = 17
model = Fast_RCNN(num_classes=num_classes, num_keypoints=num_keypoints)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
train_model(train_loader, model, optimizer, num_epochs=4)

# Save the trained model
model_save_path = './keypoint_model.pth'  # Specify the path to save the model
torch.save(model.state_dict(), model_save_path)