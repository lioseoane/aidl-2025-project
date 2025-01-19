from torch.utils.data import Dataset
import os
import cv2
import torch

class KeypointDataset(Dataset):
    def __init__(self, image_paths, keypoints, resize_to=128):
        """
        Initialize the KeypointDataset.

        Args:
            image_paths (list): List of image file paths.
            keypoints (list): List of keypoints corresponding to the images.
            resize_to (int): Target size for resizing the images (default: 256).
        """
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.resize_to = resize_to

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        keypoints = self.keypoints[idx]

        # Load and preprocess the image with padding
        image_path = os.path.join('data/mpii_human_pose/images/', image_filename)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original dimensions
        h, w, _ = image.shape
        scale = self.resize_to / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image while keeping aspect ratio
        resized_image = cv2.resize(image, (new_w, new_h))

        # Calculate padding
        pad_top = (self.resize_to - new_h) // 2
        pad_bottom = self.resize_to - new_h - pad_top
        pad_left = (self.resize_to - new_w) // 2
        pad_right = self.resize_to - new_w - pad_left

        # Add padding
        padded_image = cv2.copyMakeBorder(
            resized_image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Rescale keypoints
        keypoints = keypoints * [scale, scale]  # Scale to resized dimensions
        keypoints += [pad_left, pad_top]  # Adjust for padding

        # Normalize keypoints to range [0, 1] relative to the padded image size
        keypoints /= [self.resize_to, self.resize_to]

        # Normalize and convert image to tensor
        image_tensor = torch.tensor(padded_image, dtype=torch.float32).permute(2, 0, 1) / 255.0 # RGB from 0 to 255
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        return image_tensor, keypoints_tensor