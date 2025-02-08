from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.dataset import WorkoutDataset
from torchvision import transforms

def create_dataloaders(image_paths, bounding_boxes, keypoints, class_names, batch_size=32, resize_to=128, val_size=0.2, random_seed=42):
    """
    Create DataLoader instances for training and validation datasets.

    Args:
        image_paths (list): List of image file paths.
        bounding_boxes (list): List of bounding boxes for each image (normalized).
        keypoints (list): List of keypoints corresponding to the images.
        class_names (list): List of class names corresponding to the images.
        batch_size (int): The batch size for the DataLoader.
        resize_to (int): The target size for resizing the images.
        val_size (float): The proportion of the data to use for validation.
        random_seed (int): The seed for random splitting.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        class_name_to_idx (dict): Class to index mapping.
    """
    # Split the data into training and validation sets
    train_image_paths, val_image_paths, train_bounding_boxes, val_bounding_boxes, train_keypoints, val_keypoints, train_class_names, val_class_names = train_test_split(
        image_paths, bounding_boxes, keypoints, class_names, test_size=val_size, random_state=random_seed
    )

    # Transformation
    transforms_resnet50 = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize the datasets for both training and validation sets
    train_dataset = WorkoutDataset(train_image_paths, train_bounding_boxes, train_keypoints, train_class_names, resize_to=resize_to, transform=transforms_resnet50)
    val_dataset = WorkoutDataset(val_image_paths, val_bounding_boxes, val_keypoints, val_class_names, resize_to=resize_to, transform=transforms_resnet50)

    # Create the DataLoader instances for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.class_name_to_idx