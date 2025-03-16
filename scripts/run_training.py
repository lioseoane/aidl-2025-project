import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.multiprocessing as mp
from torchvision import transforms

from src.data.dataloader import create_dataloaders
from src.data.load_workout_data import load_workout_data
from src.models.heatmap_fpn_v3 import heatmap_fpn
from src.training.train import train_model

def main():
    # Set the hyperparameters
    
    # === Image Processing ===
    resize_to = [352, 352]  # Resize the input image to this size
    # Valid resize options: [224x224], [256x256], [288x288], [320x320], [352x352]
    # Most source images have a width of ~360px.

    # === Training Parameters ===
    batch_size = 24  # Batch size
    num_epochs = 250  # Number of epochs
    lr = [1e-4, 1e-3, 2e-4, 1e-3]  # Learning rate [fpn, workout class, keypoints, bbox]

    # === Loss Balancing ===
    loss_weights = [0.003, 0.9, 0.097]  # Classification, keypoint loss and bounding box weights respectively
    
    # === Mixed Precision ===
    autocast_enabled = True  # Enable automatic mixed precision

    # === Task-Specific ===
    num_keypoints = 17  # Number of keypoints
    pck_thresholds = [0.01, 0.05, 0.1]  # PCK thresholds (for evaluation metrics)

    # === Testing ===
    test_mode = False  # Test arquitecture mode
    test_sampling = 0.05 # Fraction of the data to sample (if test_mode is enabled)

    # Load the workout data
    keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()
    num_classes = len(set(classes_array))
    
    # === Augmentation ===
    sigmas = [
        1.2,  # Nose
        1.2,  # Left Eye
        1.2,  # Right Eye
        1.2,  # Left Ear
        1.2,  # Right Ear
        2.4,  # Left Shoulder
        2.4,  # Right Shoulder
        2.2,  # Left Elbow
        2.2,  # Right Elbow
        2.2,  # Left Wrist
        2.2,  # Right Wrist
        2.4,  # Left Hip
        2.4,  # Right Hip
        2.2,  # Left Knee
        2.2,  # Right Knee
        2.2,  # Left Ankle
        2.2   # Right Ankle
    ]

    apply_flip = True  # Apply horizontal flip augmentation

    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),                 
        transforms.RandomGrayscale(p=0.1),     
        transforms.ToTensor(),      
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataloaders
    train_loader, val_loader, class_name_to_idx = create_dataloaders(
        images_array, bounding_boxes_array, keypoints_array,
        classes_array, batch_size=batch_size, resize_to=resize_to,
        transforms=[train_transforms, val_transforms], heatmap_size=resize_to,
        sigmas=sigmas, apply_flip=apply_flip
    )

    # Limit the training data to 1%. Test any architecture across the whole environment
    if test_mode:
        train_loader = torch.utils.data.Subset(train_loader.dataset, range(int(len(train_loader.dataset) * test_sampling)))
        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
        
        val_loader = torch.utils.data.Subset(val_loader.dataset, range(int(len(val_loader.dataset) * test_sampling)))
        val_loader = torch.utils.data.DataLoader(val_loader, batch_size=batch_size, shuffle=False)

    # Initialize model
    backbone_type = 'resnet50'
    model_tag = f'{backbone_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    model = heatmap_fpn(num_classes=num_classes, num_keypoints=num_keypoints, backbone=backbone_type)

    # Specify the path to save the model
    model_save_path = f'./{model_tag}.pth' 

    # Train the model
    train_model(
        train_loader=train_loader,
        model=model,
        class_name_to_idx=class_name_to_idx,
        num_epochs=num_epochs,
        val_loader=val_loader,
        model_tag=model_tag,
        autocast_enabled=autocast_enabled,
        loss_weights=loss_weights,
        resize_to=resize_to,
        lr=lr,
        pck_thresholds=pck_thresholds
    )

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    mp.freeze_support()
    main()
