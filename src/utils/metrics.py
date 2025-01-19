import numpy as np

def calculate_pck(pred_keypoints, true_keypoints, threshold=5):
    """
    Compute the Percentage of Correct Keypoints (PCK) within a given threshold.
    """
    correct_keypoints = 0
    total_keypoints = pred_keypoints.shape[0]  # Assuming pred_keypoints has shape (num_keypoints, 2)
    
    for pred, true in zip(pred_keypoints, true_keypoints):
        dist = np.linalg.norm(pred - true, axis=1)  # Euclidean distance between predicted and true keypoints
        correct_keypoints += np.sum(dist <= threshold)  # Count keypoints within the threshold

    pck = pck = correct_keypoints / (total_keypoints * len(pred_keypoints)) * 100  # Percentage across the batch
    
    return pck