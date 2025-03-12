from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.dataset import WorkoutDataset

def create_dataloaders(image_paths, 
                       bounding_boxes, 
                       keypoints, 
                       class_names, 
                       batch_size=32, 
                       resize_to=[224, 224], 
                       val_size=0.2, 
                       random_seed=42, 
                       transforms=None,
                       heatmap_size=[224, 224],
                       sigma = 2,
                       apply_flip = True):

    # Split the data into training and validation sets
    train_image_paths, val_image_paths, train_bounding_boxes, val_bounding_boxes, train_keypoints, val_keypoints, train_class_names, val_class_names = train_test_split(
        image_paths, bounding_boxes, keypoints, class_names, test_size=val_size, random_state=random_seed
    )

    # Initialize the datasets for both training and validation sets
    train_dataset = WorkoutDataset(train_image_paths, train_bounding_boxes, train_keypoints, train_class_names, resize_to=resize_to, 
                                   transform=transforms, heatmap_size=heatmap_size, sigma=sigma, apply_flip=apply_flip)
    val_dataset = WorkoutDataset(val_image_paths, val_bounding_boxes, val_keypoints, val_class_names, resize_to=resize_to, 
                                 transform=transforms, class_name_to_idx=train_dataset.class_name_to_idx, heatmap_size=heatmap_size, 
                                 sigma=sigma, apply_flip=apply_flip)

    # Create the DataLoader instances for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, train_dataset.class_name_to_idx