import torch

def calculate_classification_accuracy(predicted_labels, true_labels, num_classes):
    # Get the predicted class labels
    predicted_labels = torch.argmax(predicted_labels, dim=1)
    true_labels = torch.argmax(true_labels, dim=1)

    # Initialize per-class TP, FP, FN
    TP = torch.zeros(num_classes, dtype=torch.float32, device=predicted_labels.device)
    FP = torch.zeros(num_classes, dtype=torch.float32, device=predicted_labels.device)
    FN = torch.zeros(num_classes, dtype=torch.float32, device=predicted_labels.device)

    # Compute per-class TP, FP, FN
    for cls in range(num_classes):
        TP[cls] = ((predicted_labels == cls) & (true_labels == cls)).sum()
        FP[cls] = ((predicted_labels == cls) & (true_labels != cls)).sum()
        FN[cls] = ((predicted_labels != cls) & (true_labels == cls)).sum()

    return TP, FP, FN

def calculate_keypoint_accuracy(predicted_keypoints, true_keypoints, thresholds=[0.01, 0.05, 0.1], image_size=None):
    # Extract xy coords and confidence
    pred_xy = predicted_keypoints[:, :, :2]  
    pred_conf = predicted_keypoints[:, :, 2]
    true_xy = true_keypoints[:, :, :2]
    true_conf = true_keypoints[:, :, 2]

    # Create visibility masks
    true_visible_mask = true_conf >= 0.5
    pred_conf_mask = pred_conf >= 0.5
    valid_mask = true_visible_mask & pred_conf_mask

    original_thresholds = thresholds.copy() # Save original thresholds

    # If working in pixel space, scale normalized coordinates
    if image_size is not None:
        width, height = image_size
        pred_xy = pred_xy * torch.tensor([width, height], device=pred_xy.device)
        true_xy = true_xy * torch.tensor([width, height], device=true_xy.device)
        thresholds = [t * max(width, height) for t in thresholds]  

    # Compute Euclidean distances between predictions and ground truths
    distances = torch.norm(pred_xy - true_xy, dim=2) 

    # Initialize correct counts for each threshold
    correct_per_threshold = {}

    # Iterate over both original and scaled thresholds
    for orig_thresh, scaled_thresh in zip(original_thresholds, thresholds):
        # Count correct keypoints: distance < threshold and both visible + confident
        correct = ((distances < scaled_thresh) & valid_mask).sum().item()
        correct_per_threshold[orig_thresh] = correct

    # Sum distances for MPJPE
    distance_sum = (distances * true_visible_mask.float()).sum().item()

     # Count of keypoints where GT is visible
    total_visible_keypoints = true_visible_mask.sum().item()

    # Count of predicted keypoints where pred_conf ≥ 0.5
    predicted_confident_keypoints = pred_conf_mask.sum().item()

    return correct_per_threshold, distance_sum, total_visible_keypoints, predicted_confident_keypoints

def calculate_bbox_accuracy(predicted_boxes, true_boxes, threshold=0.8):

    if predicted_boxes.ndim == 1:
        predicted_boxes = predicted_boxes.unsqueeze(0)

    if true_boxes.ndim == 1:
        true_boxes = true_boxes.unsqueeze(0)

    device = predicted_boxes.device
    true_boxes = true_boxes.to(device)
    
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