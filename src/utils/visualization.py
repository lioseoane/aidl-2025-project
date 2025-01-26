import cv2
import numpy as np

def visualize_keypoints(image, predicted_keypoints, true_keypoints, img_width, img_height, 
                        predicted_bbox=None, true_bbox=None):

    # Convert image to BGR for OpenCV
    image = np.transpose(image, (1, 2, 0))  # From CHW to HWC format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Denormalize the keypoints (assuming they are normalized in the range [0, 1])
    predicted_keypoints[:, :2] = predicted_keypoints[:, :2] * np.array([img_width, img_height])  # Denormalize to image dimensions
    true_keypoints[:, :2] = true_keypoints[:, :2] * np.array([img_width, img_height])  # Denormalize true keypoints

    # Adjust bounding boxes to the image size and add padding
    if predicted_bbox is not None:
        # Rescale bounding box to original size
        predicted_bbox[:2] = predicted_bbox[:2] * np.array([img_width, img_height])  # x_min, y_min
        predicted_bbox[2:] = predicted_bbox[2:] * np.array([img_width, img_height])  # x_max, y_max

        # Draw predicted bounding box (in red)
        cv2.rectangle(image, 
                      (int(predicted_bbox[0]), int(predicted_bbox[1])), 
                      (int(predicted_bbox[2]), int(predicted_bbox[3])), 
                      (0, 0, 255), 2)  # Red for predicted bbox

    if true_bbox is not None:
        true_bbox = true_bbox.squeeze()
        # Rescale bounding box to original size
        true_bbox[:2] = true_bbox[:2] * np.array([img_width, img_height])  # x_min, y_min
        true_bbox[2:] = true_bbox[2:] * np.array([img_width, img_height])  # x_max, y_max

        # Draw true bounding box (in green)
        cv2.rectangle(image, 
                      (int(true_bbox[0]), int(true_bbox[1])), 
                      (int(true_bbox[2]), int(true_bbox[3])), 
                      (0, 255, 0), 2)  # Green for true bbox

    # Draw true keypoints (in green)
    for keypoint in true_keypoints:
        x, y = keypoint[:2]  # Only take the x and y coordinates (ignoring any third value)
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green for true keypoints
    
    # Draw predicted keypoints (in red)
    for keypoint in predicted_keypoints:
        x, y = keypoint[:2]  # Only take the x and y coordinates (ignoring any third value)
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red for predicted keypoints

    # Convert image back to CHW format for TensorBoard
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))  # From HWC to CHW format
    
    return image
