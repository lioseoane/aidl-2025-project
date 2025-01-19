import cv2
import numpy as np

def visualize_keypoints(image, predicted_keypoints, true_keypoints, img_width, img_height):
    """
    Visualize the predicted and true keypoints on the image for TensorBoard logging.
    Denormalize the keypoints to the original image size.
    """
    # Convert image to BGR for OpenCV
    image = np.transpose(image, (1, 2, 0))  # From CHW to HWC format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Denormalize the keypoints (assuming they are normalized in the range [0, 1])
    predicted_keypoints = predicted_keypoints * np.array([img_width, img_height])  # Denormalize to image dimensions
    true_keypoints = true_keypoints * np.array([img_width, img_height])  # Denormalize true keypoints

    # Draw true keypoints (in green)
    for (x, y) in true_keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # Draw predicted keypoints (in red)
    for (x, y) in predicted_keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Convert image back to CHW format for TensorBoard
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))  # From HWC to CHW format
    
    return image