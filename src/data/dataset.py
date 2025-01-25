from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np

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
    

class NewKeypointDataset(Dataset):
    def __init__(self, image_paths, bounding_boxes, keypoints, class_names, resize_to=128):

        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.keypoints = keypoints
        self.class_names = class_names
        self.resize_to = resize_to

        # Create a class-to-index mapping
        self.class_name_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(class_names)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        bbox = self.bounding_boxes[idx]
        keypoints = self.keypoints[idx]
        class_name = self.class_names[idx]

        # Convert class name to class index (numerical label)
        class_label = self.class_name_to_idx[class_name]

        if class_name is None or class_name not in self.class_name_to_idx:
            print(f"Warning: Invalid or missing class_name at index {idx}: {class_name}")
            class_label = -1  # Set to a default invalid class index
        else:
            # Convert class name to class index (numerical label)
            class_label = self.class_name_to_idx[class_name]

        # Load the image
        image = cv2.imread(image_filename)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_filename}")
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

        # Normalize bounding box based on the resized and padded image
        bbox = np.array(bbox)
        bbox = bbox * [scale, scale, scale, scale]
        bbox += [pad_left, pad_top, pad_left, pad_top]
        bbox /= [self.resize_to, self.resize_to, self.resize_to, self.resize_to]

        # Normalize keypoints to resized and padded dimensions
        keypoints = np.array(keypoints)
        keypoints *= [scale, scale]
        keypoints += [pad_left, pad_top]
        keypoints /= [self.resize_to, self.resize_to]

        # Add visibility flag: If keypoint is (0, 0), set visibility to 0, otherwise set it to 1
        visibility = np.ones((keypoints.shape[0], 1))  # Start with visibility flag 1 for all
        visibility[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = 0  # Set visibility to 0 if (x, y) == (0, 0)

        # Stack the visibility flag with the keypoints
        keypoints = np.column_stack([keypoints, visibility])  # Add visibility as the third dimension

        # Convert to tensors
        image_tensor = torch.tensor(padded_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        class_label_tensor = torch.tensor([class_label], dtype=torch.long)

        #print(f"DEBUG: bbox_tensor = {bbox_tensor}, type = {type(bbox_tensor)}")
        #print(f"DEBUG: class_label_tensor = {class_label_tensor}, type = {type(class_label_tensor)}")
        #print(f"DEBUG: keypoints_tensor = {keypoints_tensor}, type = {type(keypoints_tensor)}")


        # Create the target dictionary
        target = {}
        target['boxes'] = bbox_tensor
        target['labels'] = class_label_tensor
        target['keypoints'] = keypoints_tensor

        #for k, v in target.items():
            #if not isinstance(v, torch.Tensor):
                #raise ValueError(f"Invalid value for key '{k}': {v} (type: {type(v)})")

        #print("Keypoints values:", target['keypoints'])
        #print("boxes values:", target['boxes'])
        #print("labels values:", target['labels'])
        #print (target)   

        return image_tensor, target
