import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataloader import create_dataloaders
from src.data.load_workout_data import load_workout_data
from src.models.resnet_with_heads import resnet_with_heads
from src.training.train import train_model
import torch

# Load data (images and keypoints)2
keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

# Create dataloaders
train_loader, val_loader, class_name_to_idx = create_dataloaders(images_array, bounding_boxes_array, keypoints_array, 
                                                                 classes_array, batch_size=4, resize_to=[224, 224])

# Limit the training data to 1% --> To test any arquitecture across the whole enviroment
#train_loader = torch.utils.data.Subset(train_loader.dataset, range(int(len(train_loader.dataset) * 0.02)))
#train_loader = torch.utils.data.DataLoader(train_loader, batch_size=4, shuffle=True)  # Re-create DataLoader for the subset
#val_loader = torch.utils.data.Subset(val_loader.dataset, range(int(len(val_loader.dataset) * 0.02)))
#val_loader = torch.utils.data.DataLoader(val_loader, batch_size=4, shuffle=True)  # Re-create DataLoader for the subset


# Initialize model, optimizer, and loss function
num_classes = len(set(classes_array)) 
num_keypoints = 17
model_type = 'keypoint-rcnn'
model = resnet_with_heads(num_classes=num_classes, num_keypoints=num_keypoints, model=model_type)

# Train the model
train_model(train_loader, model, class_name_to_idx, num_epochs=50, val_loader=val_loader)

# Save the trained model
model_save_path = f'./{model_type}.pth'  # Specify the path to save the model
torch.save(model.state_dict(), model_save_path)
