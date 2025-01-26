from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class WorkoutDataset(Dataset):
    def __init__(self, image_paths, bounding_boxes, keypoints, class_names, resize_to=128):

        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.keypoints = keypoints
        self.class_names = class_names
        self.resize_to = resize_to

        # Create a class-to-index mapping
        self.class_name_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(class_names)))}
        self.num_classes = len(self.class_name_to_idx)

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
        bbox = bbox * [new_w, new_h, new_w, new_h]  # Rescale to the resized image size
        bbox += [pad_left, pad_top, pad_left, pad_top]
        bbox = bbox / [self.resize_to, self.resize_to, self.resize_to, self.resize_to]

        # Normalize keypoints to resized and padded dimensions
        keypoints = np.array(keypoints)
        keypoints *= [new_w, new_h]
        keypoints += [pad_left, pad_top]
        keypoints /= [self.resize_to, self.resize_to]

        # Add visibility flag: If keypoint is (0, 0), set visibility to 0, otherwise set it to 1
        visibility = np.ones((keypoints.shape[0], 1))  # Start with visibility flag 1 for all
        visibility[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = 0  # Set visibility to 0 if (x, y) == (0, 0)

        # Stack the visibility flag with the keypoints
        keypoints = np.column_stack([keypoints, visibility])  # Add visibility as the third dimension

        # Convert to tensors
        image_tensor = torch.tensor(padded_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0) 
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        class_label_tensor = torch.tensor(class_label, dtype=torch.int64).unsqueeze(0) 

        #class_label_one_hot = torch.zeros(self.num_classes, dtype=torch.int64)
        #if class_label >= 0:  # Only assign if the class label is valid
            #class_label_one_hot[class_label] = 1


        # Create the target dictionary
        target = {}
        target['boxes'] = bbox_tensor
        target['labels'] = class_label_tensor  
        target['keypoints'] = keypoints_tensor

        return image_tensor, target
