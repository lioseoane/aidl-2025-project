import numpy as np
import torch

def calculate_classification_accuracy(predicted_labels, true_labels):

    # Get the predicted class labels
    predicted_labels = torch.argmax(predicted_labels, dim=1)
    true_labels = torch.argmax(true_labels, dim=1)

    # Count correct predictions
    correct = (predicted_labels == true_labels).sum().item()  
    
    # Calculate accuracy
    accuracy = correct / len(true_labels)  
    return accuracy

def calculate_keypoint_accuracy(predicted_keypoints, true_keypoints, threshold=0.03):

    # Extract (x, y) and visibility flag
    pred_xy = predicted_keypoints[:, :, :2]
    true_xy = true_keypoints[:, :, :2]  # Shape: (num_keypoints, 2)
    visibility = true_keypoints[:, :, 2]  # Shape: (num_keypoints,)

    # Calculate Euclidean distance between predicted and true keypoints
    distances = torch.norm(pred_xy - true_xy, dim=-1)

    # Count correct keypoints within the threshold
    visible_mask = visibility > 0  # Boolean mask where visibility == 1

    correct_keypoints = ((distances < threshold) & visible_mask).sum().item()
    total_visible_keypoints = visible_mask.sum().item()

    # Calculate accuracy
    accuracy = correct_keypoints / total_visible_keypoints if total_visible_keypoints > 0 else 0.0
    return accuracy

def calculate_bbox_accuracy(predicted_boxes, true_boxes, threshold=0.8):

    # [batch_size, 1, 4] -> [batch_size, 4]
    #predicted_boxes = predicted_boxes.squeeze(1)
    #true_boxes = true_boxes.squeeze(1)

    # Get the coordinates of bounding boxes
    x1 = torch.max(predicted_boxes[:, 0], true_boxes[:, 0])
    y1 = torch.max(predicted_boxes[:, 1], true_boxes[:, 1])
    x2 = torch.min(predicted_boxes[:, 2], true_boxes[:, 2])
    y2 = torch.min(predicted_boxes[:, 3], true_boxes[:, 3])

    # Calculate intersection and union areas
    intersection_area = torch.max(x2 - x1, torch.tensor(0.0, device=predicted_boxes.device)) * \
                        torch.max(y2 - y1, torch.tensor(0.0, device=predicted_boxes.device))
    predicted_area = (predicted_boxes[:, 2] - predicted_boxes[:, 0]) * (predicted_boxes[:, 3] - predicted_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

    # Calculate IoU
    union_area = predicted_area + true_area - intersection_area
    iou = intersection_area / union_area

    # Count boxes with IoU above the threshold
    correct_boxes = (iou > threshold).sum().item()
    total_boxes = len(iou)

     # Calculate accuracy
    accuracy = correct_boxes / total_boxes
    return accuracy