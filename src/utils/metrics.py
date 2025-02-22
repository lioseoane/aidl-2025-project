import numpy as np
import torch

def calculate_classification_accuracy(predicted_labels, true_labels, num_classes):
    # Get the predicted class labels
    predicted_labels = torch.argmax(predicted_labels, dim=1)
    true_labels = torch.argmax(true_labels, dim=1)

    # Count correct predictions
    correct = (predicted_labels == true_labels).sum().item()  
    accuracy = correct / len(true_labels)  

    # Initialize variables for overall precision and recall
    TP = ((predicted_labels == true_labels) & (true_labels != -1)).sum().item()  # True Positives (ignoring padding)
    FP = ((predicted_labels != true_labels) & (true_labels != -1)).sum().item()  # False Positives (ignoring padding)
    FN = ((predicted_labels != true_labels) & (true_labels != -1)).sum().item()  # False Negatives (ignoring padding)
    
    return accuracy, TP, FP, FN

def calculate_keypoint_accuracy(predicted_keypoints, true_keypoints, threshold=0.01):
    batch_size, num_keypoints, last_dim = predicted_keypoints.shape

    if last_dim == 2:
        # Already (x, y)
        pred_xy = predicted_keypoints
        true_xy = true_keypoints

        # Assume all keypoints are visible
        visible_mask = torch.ones(batch_size, num_keypoints, dtype=torch.bool)

    elif last_dim == 3:
        # Extract (x, y)
        pred_xy = predicted_keypoints[:, :, :2]
        true_xy = true_keypoints[:, :, :2]

        # Extract visibility if it exists
        if true_keypoints.shape[-1] == 3:
            visibility = true_keypoints[:, :, 2] > 0  # Boolean mask where visibility > 0
        else:
            # If no visibility information, assume all keypoints are visible
            visibility = torch.ones(batch_size, num_keypoints, dtype=torch.bool)

        visible_mask = visibility

    # Compute Euclidean distances
    distances = torch.norm(pred_xy - true_xy, dim=-1)
    
    # Move to CUDA 
    device = distances.device
    visible_mask = visible_mask.to(device)

    # Count correct keypoints within the threshold
    correct_keypoints = ((distances < threshold) & visible_mask).sum().item()
    total_visible_keypoints = visible_mask.sum().item()

    # Calculate accuracy
    accuracy = correct_keypoints / total_visible_keypoints if total_visible_keypoints > 0 else 0.0
    return accuracy

def calculate_bbox_accuracy(predicted_boxes, true_boxes, threshold=0.7):
    # Get the intersection coordinates
    x1 = torch.max(predicted_boxes[:, 0], true_boxes[:, 0])  # Left x-coordinate
    y1 = torch.max(predicted_boxes[:, 1], true_boxes[:, 1])  # Top y-coordinate
    x2 = torch.min(predicted_boxes[:, 2], true_boxes[:, 2])  # Right x-coordinate
    y2 = torch.min(predicted_boxes[:, 3], true_boxes[:, 3])  # Bottom y-coordinate

    # Ensure the intersection dimensions are valid (non-negative)
    inter_width = torch.max(x2 - x1, torch.tensor(0.0, device=predicted_boxes.device))
    inter_height = torch.max(y2 - y1, torch.tensor(0.0, device=predicted_boxes.device))

    # Intersection area
    intersection_area = inter_width * inter_height

    # Predicted and true bounding box areas
    predicted_area = (predicted_boxes[:, 2] - predicted_boxes[:, 0]) * (predicted_boxes[:, 3] - predicted_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

    # Union area
    union_area = predicted_area + true_area - intersection_area

    # IoU (Intersection over Union)
    iou = intersection_area / (union_area + 1e-6)  # Add a small epsilon to avoid division by zero

    # Count boxes with IoU greater than the threshold
    correct_boxes = (iou > threshold).sum().item()
    total_boxes = len(iou)

    # Calculate accuracy
    accuracy = correct_boxes / total_boxes if total_boxes > 0 else 0.0  # Avoid division by zero
    return accuracy