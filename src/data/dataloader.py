from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.dataset import KeypointDataset  # Ensure this import is correct
import numpy as np

def create_dataloaders(images_array, keypoints_array, batch_size=32, resize_to=128, val_size=0.2, random_seed=42):
    """
    Create DataLoader instances for training and validation datasets.

    Args:
        images_array (np.ndarray): Array of image file paths.
        keypoints_array (np.ndarray): Array of keypoints corresponding to the images.
        batch_size (int): The batch size for the DataLoader.
        resize_to (int): The target size for resizing the images.
        val_size (float): The proportion of the data to use for validation.
        random_seed (int): The seed for random splitting.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
    """
    # Split the data into training and validation sets
    train_images, val_images, train_keypoints, val_keypoints = train_test_split(
        images_array, keypoints_array, test_size=val_size, random_state=random_seed
    )

    # Initialize the datasets for both training and validation sets
    train_dataset = KeypointDataset(train_images, train_keypoints, resize_to=resize_to)
    val_dataset = KeypointDataset(val_images, val_keypoints, resize_to=resize_to)

    # Create the DataLoader instances for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
