import torch
import torch.nn.functional as F

def generate_heatmaps(keypoints, output_size=(224, 224), sigmas=None):
    num_keypoints = keypoints.shape[0]
    height, width = output_size

    # Create coordinate grid
    y_range = torch.arange(0, height, dtype=torch.float).view(height, 1).expand(height, width)
    x_range = torch.arange(0, width, dtype=torch.float).view(1, width).expand(height, width)

    # Initialize heatmaps
    heatmaps = torch.zeros((num_keypoints, height, width), dtype=torch.float)

    if sigmas is None:
        sigmas = [2.0 for _ in range(num_keypoints)]  # fallback sigma if none provided

    for i in range(num_keypoints):
        x_norm, y_norm, conf = keypoints[i]

        if conf >= 0.5:  # Only create heatmap for valid keypoints
            # Convert normalized coordinates to pixel coordinates
            x = x_norm * (width - 1)
            y = y_norm * (height - 1)

            # Fetch sigma for this keypoint
            sigma = sigmas[i]

            # Compute Gaussian heatmap
            heatmap = torch.exp(-((x_range - x) ** 2 + (y_range - y) ** 2) / (2 * sigma ** 2))
            heatmap /= heatmap.max()  # Normalize to range [0,1]
            
            heatmaps[i] = heatmap
        
    return heatmaps

def extract_keypoints_with_confidence(heatmaps, refine=True):
    """
    Extract keypoints from heatmaps with sub-pixel precision and confidence scores.
    
    Args:
        heatmaps (Tensor): Shape (B, num_keypoints, H, W)
        refine (bool): If True, refine keypoint locations using local offset estimation.
    
    Returns:
        keypoints (Tensor): Shape (B, num_keypoints, 3) -> (x, y, confidence)
    """
    B, num_keypoints, H, W = heatmaps.shape

    # Reshape heatmaps for easier indexing
    heatmaps_flat = heatmaps.view(B, num_keypoints, -1)  # Shape: (B, num_keypoints, H*W)

    # Find the max value and corresponding index (integer pixel location)
    confidences, indices = torch.max(heatmaps_flat, dim=2)  # (B, num_keypoints)

    # Convert indices into (x, y) coordinates
    x_coords = (indices % W).float()
    y_coords = (indices // W).float()

    if refine:
        # Get 3x3 local patches around detected keypoints
        heatmaps_padded = F.pad(heatmaps, (1, 1, 1, 1), mode='replicate')  # Padding for edges
        x_coords_int = x_coords.long() + 1  # Offset due to padding
        y_coords_int = y_coords.long() + 1

        refined_x = torch.zeros_like(x_coords)
        refined_y = torch.zeros_like(y_coords)

        for b in range(B):
            for k in range(num_keypoints):
                # Extract local 3x3 patch
                patch = heatmaps_padded[b, k, y_coords_int[b, k]-1:y_coords_int[b, k]+2, 
                                                x_coords_int[b, k]-1:x_coords_int[b, k]+2]
                
                # Compute local gradients
                dx = (patch[:, 2] - patch[:, 0]) / 2.0  # Gradient in x-direction
                dy = (patch[2, :] - patch[0, :]) / 2.0  # Gradient in y-direction
                
                # Sub-pixel correction: Offset keypoints using the gradients
                refined_x[b, k] = x_coords[b, k] + dx.mean()
                refined_y[b, k] = y_coords[b, k] + dy.mean()
        
        x_coords = refined_x
        y_coords = refined_y

    # Normalize coordinates to [0, 1] range
    x_norm = x_coords / (W - 1)
    y_norm = y_coords / (H - 1)

    # Stack (x, y, confidence)
    keypoints = torch.stack([x_norm, y_norm, confidences], dim=2)  # Shape: (B, num_keypoints, 3)

    return keypoints

def generate_bbox_heatmaps(box, heatmap_size=(224, 224)):
    """
    box: [x1, y1, x2, y2] in normalized [0, 1] coords
    """
    h, w = heatmap_size
    heatmap = torch.zeros((h, w), dtype=torch.float32)

    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)

    heatmap[y1:y2+1, x1:x2+1] = 1.0
    return heatmap

def extract_bbox_from_heatmaps(bbox_heatmaps, threshold=0.8):
    """
    Args:
        bbox_heatmaps: Tensor of shape [batch_size, H, W] (binary heatmap of bbox regions)
        threshold: Threshold for binarization
    Returns:
        bboxes: Tensor of shape [batch_size, 4] (normalized x1, y1, x2, y2)
    """
    B, H, W = bbox_heatmaps.shape

    bboxes = []

    for b in range(B):
        heatmap = bbox_heatmaps[b]  # shape [H, W]
        mask = (heatmap >= threshold).nonzero(as_tuple=False)

        if mask.size(0) == 0:
            bbox = torch.tensor([0, 0, 1, 1], dtype=torch.float32, device=heatmap.device)
        else:
            y_coords = mask[:, 0]
            x_coords = mask[:, 1]

            x1 = x_coords.min().item() / W
            x2 = x_coords.max().item() / W
            y1 = y_coords.min().item() / H
            y2 = y_coords.max().item() / H

            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=heatmap.device)

        bboxes.append(bbox)

    return torch.stack(bboxes, dim=0)  # [B, 4]