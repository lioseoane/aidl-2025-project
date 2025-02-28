import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataloader import create_dataloaders
from src.data.load_workout_data import load_workout_data
from src.models.baseline_heatmap import baseline_heatmap
from src.training.train import train_model
import torch
from datetime import datetime
from torchvision import transforms

# Transformations
#transforms_resnet50 = transforms.Compose([
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])

# Load data (images and keypoints)2
keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

# Create dataloaders
train_loader, val_loader, class_name_to_idx = create_dataloaders(images_array, bounding_boxes_array, keypoints_array, 
                                                                 classes_array, batch_size=16, resize_to=[224, 224], 
                                                                 transforms=None, heatmap_size=[224, 224],
                                                                 sigma=2)

# Limit the training data to 1% --> To test any arquitecture across the whole enviroment
#train_loader = torch.utils.data.Subset(train_loader.dataset, range(int(len(train_loader.dataset) * 0.1)))
#train_loader = torch.utils.data.DataLoader(train_loader, batch_size=16, shuffle=True)  # Re-create DataLoader for the subset
#val_loader = torch.utils.data.Subset(val_loader.dataset, range(int(len(val_loader.dataset) * 0.1)))
#val_loader = torch.utils.data.DataLoader(val_loader, batch_size=16, shuffle=False)  # Re-create DataLoader for the subset

# Initialize model, optimizer, and loss function
num_classes = len(set(classes_array)) 
num_keypoints = 17
backbone_type = 'resnet50'
model = baseline_heatmap(num_classes=num_classes, num_keypoints=num_keypoints, backbone=backbone_type)

# Train the model
train_model(train_loader, model, class_name_to_idx, num_epochs=75, val_loader=val_loader)
#train_model(train_loader, model, class_name_to_idx, num_epochs=50, val_loader=None)

# Save the trained model

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = f'./{backbone_type}_{timestamp}.pth'  # Specify the path to save the model
torch.save(model.state_dict(), model_save_path)