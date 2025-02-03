from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class WorkoutDataset(Dataset):
    def __init__(self, image_paths, bounding_boxes, keypoints, class_names, resize_to=[480,360]):

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
        
        # Load the image
        image = cv2.imread(image_filename)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original dimensions
        h, w, _ = image.shape
        target_w, target_h = self.resize_to

        scale_w = target_w / float(w)
        scale_h = target_h / float(h)

        bbox = np.array(bbox)
        keypoints = np.array(keypoints)

         # Add visibility flag: If keypoint is (0, 0), set visibility to 0, otherwise set it to 1
        visibility = np.ones((keypoints.shape[0], 1))  # Start with visibility flag 1 for all
        visibility[(keypoints[:, 0] == 0) | (keypoints[:, 1] == 0)] = 0  # Set visibility to 0 if (x, y) == (0, 0)

        if scale_w == 1.0 and scale_h == 1.0:
            padded_image = image
        else:
            scale = min(scale_w, scale_h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

            if  scale_w != scale_h:
                # Rescale to the resized image size
                bbox = bbox * [new_w, new_h, new_w, new_h]
                keypoints *= [new_w, new_h]

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
                keypoints += [pad_left, pad_top]

                # Normalized to 0,1
                bbox = bbox / [target_w, target_h, target_w, target_h]
                keypoints /= [target_w, target_h]

                # Set keypoints to (0, 0) if visibility is 0 after padding and normalization (for consistency)
                for i in range(len(visibility)):
                    if visibility[i] == 0:  # If visibility is 0
                        keypoints[i] = [0.0, 0.0]


        # Stack the visibility flag with the keypoints
        keypoints = np.column_stack([keypoints, visibility])  # Add visibility as the third dimension

        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        class_label_one_hot = torch.zeros(self.num_classes, dtype=torch.int64)
        if class_label >= 0:  # Only assign if the class label is valid
            class_label_one_hot[class_label] = 1


        # Create the target dictionary
        target = {}
        target['bbox'] = bbox_tensor
        target['workout_label'] = class_label_one_hot  
        target['keypoints'] = keypoints_tensor
        target['filename'] = image_filename
        target['workout_label_name'] = class_name

        return image_tensor, target
