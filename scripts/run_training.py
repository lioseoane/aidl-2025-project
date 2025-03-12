import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataloader import create_dataloaders
from src.data.load_workout_data import load_workout_data
from src.models.heatmap_fpn_v3 import heatmap_fpn
from src.training.train import train_model
import torch
from datetime import datetime

# Set the hyperparameters
resize_to = [224, 224] # Resize the input image to this size
batch_size = 16 # Batch size
sigma = 1.5 # Sigma for the gaussian kernel
apply_flip = True # Apply flip augmentation
num_epochs = 250 # Number of epochs
loss_weights = [0.0005, 0.69975, 0.29975] # Classification, keypoint loss and bounding box weights respectively
autocast_enabled = True # Enable automatic mixed precision
num_keypoints = 17 # Number of keypoints
lr = 1e-4 # Learning rate
pck_thresholds = [0.01, 0.05, 0.1] # PCK thresholds
test_mode = False # Test mode

# Load the workout data
keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()
num_classes = len(set(classes_array))

# Create dataloaders
train_loader, val_loader, class_name_to_idx = create_dataloaders(images_array, bounding_boxes_array, keypoints_array, 
                                                                 classes_array, batch_size=batch_size, resize_to=resize_to, 
                                                                 transforms=None, heatmap_size=resize_to,
                                                                 sigma=sigma, apply_flip=apply_flip)

# Limit the training data to 1% --> Test any arquitecture across the whole enviroment
if test_mode:
    train_loader = torch.utils.data.Subset(train_loader.dataset, range(int(len(train_loader.dataset) * 0.05)))
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=16, shuffle=True)  # Re-create DataLoader for the subset
    val_loader = torch.utils.data.Subset(val_loader.dataset, range(int(len(val_loader.dataset) * 0.05)))
    val_loader = torch.utils.data.DataLoader(val_loader, batch_size=16, shuffle=False)  # Re-create DataLoader for the subset

# Initialize model
backbone_type = 'resnet50'
model_tag = f'{backbone_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
model = heatmap_fpn(num_classes=num_classes, num_keypoints=num_keypoints, backbone=backbone_type)

# Specify the path to save the model
model_save_path = f'./{model_tag}.pth' 

# Train the model
train_model(train_loader, model, class_name_to_idx, num_epochs=num_epochs, val_loader=val_loader, model_tag=model_tag, 
            autocast_enabled=autocast_enabled, loss_weights=loss_weights, resize_to=resize_to, lr=lr, pck_thresholds=pck_thresholds)

# Save the trained model
torch.save(model.state_dict(), model_save_path)