from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from PIL import Image

class WorkoutDataset(Dataset):
    def __init__(self, 
                 image_paths, 
                 bounding_boxes, 
                 keypoints, 
                 class_names, 
                 resize_to=[256, 256], 
                 transform=None, 
                 class_name_to_idx=None, 
                 heatmap_size=[64, 64]):
        
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.keypoints = keypoints
        self.class_names = class_names
        self.resize_to = resize_to

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
        image = Image.open(image_filename).convert("RGB")
        #image = image.resize(self.resize_to)
        image = np.array(image)

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

        if scale_w == 1.0 and scale_h == 1.0:
            pass
        else:
            scale = min(scale_w, scale_h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

            if  scale_w != scale_h:
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

        
        bbox_denorm = bbox * [target_w, target_h, target_w, target_h]
        keypoints_denorm_xy = keypoints_xy * [target_w, target_h]

                # Merge back confidence
        if keypoints_conf is not None:
            keypoints = np.hstack([keypoints_xy, keypoints_conf])
            keypoints_denorm = np.hstack([keypoints_denorm_xy, keypoints_conf])
        else:
            keypoints = keypoints_xy
            keypoints_denorm = keypoints_denorm_xy

        # Stack the visibility flag with the keypoints
        #keypoints = np.column_stack([keypoints, visibility])  # Add visibility as the third dimension

        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        bbox_denorm_tensor = torch.tensor(bbox_denorm, dtype=torch.float32)
        keypoints_denorm_tensor = torch.tensor(keypoints_denorm, dtype=torch.float32)

        class_label_one_hot = torch.zeros(self.num_classes, dtype=torch.int64)
        if class_label >= 0:  # Only assign if the class label is valid
            class_label_one_hot[class_label] = 1


        # Create the target dictionary
        target = {}
        target['boxes'] = bbox_tensor
        target['boxes_denorm'] = bbox_denorm_tensor
        target['labels'] = torch.tensor([1], dtype=torch.int64)
        target['workout_labels'] = class_label_one_hot  
        target['keypoints'] = keypoints_tensor
        target['keypoints_denorm'] = keypoints_denorm_tensor
        target['filenames'] = image_filename
        target['workout_label_names'] = class_name

        return image_tensor, target
