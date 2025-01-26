import numpy as np
import torch

def calculate_classification_accuracy(predicted_labels, true_labels, device='cuda'):
    # Assuming predicted_labels and true_labels are tensors of shape (batch_size,)
    # Check if the predicted labels match the true labels
    predicted_labels = torch.argmax(predicted_labels, dim=1)
    true_labels = torch.from_numpy(np.array(true_labels)).to(device)
    correct = (predicted_labels == true_labels).sum().item()  # Convert to int before summing
    accuracy = correct / len(true_labels)  # Calculate accuracy
    return accuracy

def calculate_keypoint_accuracy(predicted_keypoints, true_keypoints, threshold=0.05, device='cuda'):
    # Calculate Euclidean distance between predicted and true keypoints using PyTorch
    predicted_keypoints = torch.tensor(predicted_keypoints).to(device)
    true_keypoints = torch.from_numpy(np.array(true_keypoints)).to(device)

    distances = torch.norm(predicted_keypoints - true_keypoints, dim=-1)

    # Count correct keypoints within the threshold
    correct_keypoints = (distances < threshold).sum().item()
    total_keypoints = distances.numel()  # Use numel() to get the total number of keypoints
    
    accuracy = correct_keypoints / total_keypoints
    return accuracy

def calculate_bbox_accuracy(predicted_boxes, true_boxes, threshold=0.5, device='cuda'):
    # Calculate IoU between predicted and true bounding boxes
    predicted_boxes = predicted_boxes.clone().detach().to(device).squeeze(1)
    true_boxes = torch.stack(true_boxes).to(device).squeeze(1)

    x1 = torch.max(predicted_boxes[:, 0], true_boxes[:, 0])
    y1 = torch.max(predicted_boxes[:, 1], true_boxes[:, 1])
    x2 = torch.min(predicted_boxes[:, 2], true_boxes[:, 2])
    y2 = torch.min(predicted_boxes[:, 3], true_boxes[:, 3])

    intersection_area = torch.max(x2 - x1, torch.tensor(0.0, device=predicted_boxes.device)) * \
                        torch.max(y2 - y1, torch.tensor(0.0, device=predicted_boxes.device))
    predicted_area = (predicted_boxes[:, 2] - predicted_boxes[:, 0]) * (predicted_boxes[:, 3] - predicted_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

    union_area = predicted_area + true_area - intersection_area
    iou = intersection_area / union_area

    # Count boxes with IoU above the threshold
    correct_boxes = (iou > threshold).sum().item()
    total_boxes = len(iou)
    
    accuracy = correct_boxes / total_boxes
    return accuracy