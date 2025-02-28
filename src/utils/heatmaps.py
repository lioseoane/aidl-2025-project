import torch
import torch.nn.functional as F

import torch

def generate_heatmaps(keypoints, output_size=(224, 224), sigma=1):

    num_keypoints = keypoints.shape[0]
    height, width = output_size

    # Create coordinate grid
    y_range = torch.arange(0, height, dtype=torch.float).view(height, 1).expand(height, width)
    x_range = torch.arange(0, width, dtype=torch.float).view(1, width).expand(height, width)

    # Initialize heatmaps
    heatmaps = torch.zeros((num_keypoints, height, width), dtype=torch.float)

    for i in range(num_keypoints):
        x_norm, y_norm, conf = keypoints[i]

        if conf >= 0.5:  # Only create heatmap for valid keypoints
            # Convert normalized coordinates to pixel coordinates
            x = x_norm * (width - 1)
            y = y_norm * (height - 1)

            # Compute Gaussian heatmap
            heatmap = torch.exp(-((x_range - x) ** 2 + (y_range - y) ** 2) / (2 * sigma ** 2))
            heatmap /= heatmap.max()  # Normalize to range [0,1]
            
            heatmaps[i] = heatmap
        
    return heatmaps


import torch

def extract_keypoints_with_confidence(heatmaps):
    B, num_keypoints, H, W = heatmaps.shape

    # Normalize heatmaps before extracting peaks
    heatmaps = heatmaps - heatmaps.min()  # Shift values to avoid negative scores
    heatmaps = heatmaps / (heatmaps.max() + 1e-6)  # Normalize to [0, 1]

    # Reshape heatmaps to find the max index
    heatmaps_flat = heatmaps.view(B, num_keypoints, -1)
    heatmaps_softmax = F.softmax(heatmaps_flat, dim=2)  # Apply softmax along spatial dimension

    # Get max confidence and corresponding indices
    confidences, indices = torch.max(heatmaps_softmax, dim=2)

    # Convert flat indices to (x, y) coordinates
    x_coords = (indices % W).float()  # x coordinate
    y_coords = (indices // W).float()  # y coordinate

    # Normalize coordinates to [0, 1] range
    x_norm = x_coords / (W - 1)
    y_norm = y_coords / (H - 1)

    # Stack into (x, y, confidence)
    keypoints = torch.stack([x_norm, y_norm, confidences], dim=2)  # Shape: (B, num_keypoints, 3)

    return keypoints
