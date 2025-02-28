from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from src.utils.heatmaps import generate_heatmaps, extract_keypoints_with_confidence

class WorkoutDataset(Dataset):
    def __init__(self, 
                 image_paths, 
                 bounding_boxes, 
                 keypoints, 
                 class_names, 
                 resize_to=[224, 224], 
                 transform=None, 
                 class_name_to_idx=None, 
                 heatmap_size=[224, 224],
                 sigma = 1):
        
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.keypoints = keypoints
        self.class_names = class_names
        self.resize_to = resize_to
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Create a class-to-index mapping
        if class_name_to_idx:
            self.class_name_to_idx = class_name_to_idx
        else:
            self.class_name_to_idx = {class_name: idx for idx, class_name in enumerate(set(class_names))}

        self.num_classes = len(self.class_name_to_idx)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        bbox = self.bounding_boxes[idx]
        keypoints = self.keypoints[idx]
        class_name = self.class_names[idx]

        # Convert class name to class index (numerical label)
        class_label = self.class_name_to_idx[class_name]
        
        # Load the image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original dimensions
        h, w, _ = image.shape
        target_w, target_h = self.resize_to

        scale_w = target_w / float(w)
        scale_h = target_h / float(h)

        bbox = np.array(bbox)
        keypoints = np.array(keypoints)

        # Separate x, y and confidence
        if keypoints.shape[1] == 3:  
            keypoints_xy = keypoints[:, :2]  # Extract only x, y
            keypoints_conf = keypoints[:, 2:]  # Keep confidence separate
        else:
            keypoints_xy = keypoints  # No confidence dimension
            keypoints_conf = None

        scale = min(scale_w, scale_h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

        if scale_w != scale_h:
            # Rescale to the resized image size
            bbox = bbox * [new_w, new_h, new_w, new_h]
            keypoints_xy *= [new_w, new_h]

            # Calculate padding
            pad_top = (target_h  - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w  - new_w - pad_left

            # Add padding
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
                
            # Add padding
            bbox += [pad_left, pad_top, pad_left, pad_top]
            keypoints_xy += [pad_left, pad_top]

            # Normalized to 0,1
            bbox = bbox / [target_w, target_h, target_w, target_h]
            keypoints_xy /= [target_w, target_h]

        keypoints_xy, keypoints_conf = self.fix_keypoints(keypoints_xy, keypoints_conf)
        keypoints_conf = keypoints_conf.squeeze()  # Ensure it's 1D (17,)
        keypoints_conf = keypoints_conf.reshape(-1, 1)  # Now (17,1)
        keypoints_fixed = np.hstack([keypoints_xy, keypoints_conf])

        #bbox_denorm = bbox * [target_w, target_h, target_w, target_h]
        #keypoints_denorm_xy = keypoints_xy * [target_w, target_h]

        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32) # Normalized based bbox [0, 1]
        #rois_tensor = torch.tensor(bbox_denorm, dtype=torch.float32) # Denormalized based bbox [x, y]

        keypoints_tensor = torch.tensor(keypoints_fixed, dtype=torch.float32) # Normalized based keypoints [0, 1]
        heatmaps_tensor = generate_heatmaps(keypoints_fixed, output_size=tuple(self.heatmap_size), sigma=self.sigma)

        class_label_one_hot = torch.zeros(self.num_classes, dtype=torch.int64)
        if class_label >= 0:  # Only assign if the class label is valid
            class_label_one_hot[class_label] = 1

        # Create the target dictionary
        target = {
            'boxes': bbox_tensor,
            'labels': torch.tensor([1], dtype=torch.int64),
            'workout_labels': class_label_one_hot,
            'keypoints': keypoints_tensor,
            'heatmaps': heatmaps_tensor,
            'filenames': image_filename,
            'workout_label_names': class_name
        }

        return image_tensor, target
    

    def fix_keypoints(self, keypoints_xy, keypoints_conf):
        """
        Fix keypoint inconsistencies:
        - If confidence < 0.5, replace with interpolated value or previous frame’s keypoint.
        - Ensure all 17 keypoints are present.
        """
        for i in range(len(keypoints_xy)):
            if keypoints_conf[i] < 0.5:
                keypoints_xy[i] = np.nan  # Mark missing keypoints

        # Interpolate missing keypoints
        for j in range(17):
            valid_mask = ~np.isnan(keypoints_xy[:, 0])  # Check which keypoints are valid
            if valid_mask.sum() > 0:
                keypoints_xy[:, 0] = np.interp(
                    np.arange(len(keypoints_xy)), np.where(valid_mask)[0], keypoints_xy[valid_mask, 0]
                )
                keypoints_xy[:, 1] = np.interp(
                    np.arange(len(keypoints_xy)), np.where(valid_mask)[0], keypoints_xy[valid_mask, 1]
                )

        # Replace NaNs with default values (center of image)
        keypoints_xy = np.nan_to_num(keypoints_xy, nan=0.5)  # Default position (normalized to [0,1])

        return keypoints_xy, keypoints_conf
