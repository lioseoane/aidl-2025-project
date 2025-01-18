import torch
import cv2
import os
import numpy as np
from src.models.dummy_cnn import KeypointModel  # Assuming this is your model
import sys
import random

# Load the trained model
def load_model(model_path='keypoint_model.pth'):
    """
    Load the trained model from the specified path.
    """
    model = KeypointModel(num_keypoints=16, input_size=128)
    model.load_state_dict(torch.load(model_path))  # Load the saved model weights
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path, target_size=128):
    """
    Load and preprocess the image for prediction.
    Resize with padding to match target_size (128x128), preserving the aspect ratio.
    """
    # Load the original image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded properly
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    
    # Get original dimensions
    original_h, original_w, _ = image.shape
    scale = target_size / max(original_h, original_w)
    new_w, new_h = int(original_w * scale), int(original_h * scale)

    # Resize the image to fit within the target size while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h))

    # Calculate padding (to make the image square with 128x128 dimensions)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    # Pad the image with black borders to make it 128x128
    padded_image = cv2.copyMakeBorder(
        resized_image, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Normalize the image (0-255 to 0-1 range)
    padded_image = padded_image / 255.0

    # Convert to tensor and add batch dimension (CxHxW)
    image_tensor = torch.tensor(padded_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Return padding values to later reverse keypoint denormalization
    return image_tensor, (pad_left, pad_top, scale)

def visualize_keypoints(image_path, predicted_keypoints, padding, target_size=128):
    """
    Visualize the predicted keypoints on the original image.
    The keypoints are denormalized to the original image size.
    """
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the padding information
    pad_left, pad_top, scale = padding

    # Denormalize the keypoints to match the original image size
    denormalized_keypoints = []
    for (x, y) in predicted_keypoints:
        # Rescale the keypoints to the original image dimensions
        original_x = (x * target_size - pad_left) / scale
        original_y = (y * target_size - pad_top) / scale
        denormalized_keypoints.append((original_x, original_y))

    # Draw the keypoints on the original image
    for (x, y) in denormalized_keypoints:
        # Convert the coordinates to integers
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Display the image with keypoints
    try:
        cv2.imshow("Keypoints", image)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
        sys.exit()  # Ensure the script exits completely

if __name__ == '__main__':
    # Path to the images folder
    images_folder = 'data/mpii_human_pose/images'

    # Get a random image from the images folder
    image_files = os.listdir(images_folder)
    image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]  # Only image files
    random_image_path = os.path.join(images_folder, random.choice(image_files))

    # Load the trained model
    model = load_model('keypoint_model.pth')

    # Run inference and visualize keypoints for the random image
    image_tensor, padding = preprocess_image(random_image_path, target_size=128)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)  # model inference

    # Convert prediction to keypoints (output from the model)
    predicted_keypoints = predictions.squeeze().cpu().numpy().reshape(-1, 2)

    # Visualize the keypoints on the original image
    visualize_keypoints(random_image_path, predicted_keypoints, padding, target_size=128)