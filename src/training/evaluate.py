import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_bbox_accuracy, calculate_keypoint_accuracy
from torch.amp import autocast
import os
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps
import matplotlib.cm as cm
import numpy as np

def evaluate_model(val_loader, model, class_name_to_idx, model_tag='0', log_dir="logs/", num_epoch=0, autocast_enabled=True, 
                   loss_weights=[1, 1, 1], resize_to=[224, 224], pck_thresholds=[0.01, 0.05, 0.1]):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if CUDA is available

    # Move model to the same device as the data
    model = model.to(device)

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping
    num_classes = len(idx_to_class_name)

    # Create logs folder and initialize TensorBoard writer
    log_dir_str = f'{log_dir}/{model_tag}/val_logs'
    os.makedirs(log_dir_str, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_str)

    # Set model to evaluation mode
    model.eval()

    # Initialize accumulators for losses at the epoch level
    val_classification_loss= 0.0
    val_keypoint_loss = 0.0
    val_bbox_loss = 0.0
    val_loss = 0.0

    # Initialize accumulators for accuracy metrics at the epoch level
    total_TP = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_FP = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_FN = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_classification_correct = 0
    total_classification_count = 0
    total_bbox_correct = 0
    total_bbox_count = 0
    # Initialize totals for thresholds
    total_correct_per_threshold = {t: 0 for t in pck_thresholds}
    total_distance_sum = 0
    total_visible = 0
    total_predicted_visible = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch_idx, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):

            # Move data to the same device as the model
            images = images.to(device)

            # Move targets to the same device as the model
            new_targets = []
            for i in range(len(targets["boxes"])):  

                new_targets.append({
                    "boxes": targets["boxes"][i].to(device),  
                    "workout_labels": targets["workout_labels"][i].to(device),  
                    "keypoints": targets["heatmaps"][i].to(device), 
                    "confidences": targets["confidences"][i].to(device), 
                })

            # Forward pass and compute loss
            if autocast_enabled:
                with autocast("cuda"):
                    output = model(images) # Forward pass
                    bbox_loss, keypoints_loss, classification_loss = model.compute_losses(output, new_targets) # Losses
            else:
                output = model(images) # Forward pass
                bbox_loss, keypoints_loss, classification_loss = model.compute_losses(output, new_targets) # Losses

            # Compute total loss
            total_loss = loss_weights[0] * classification_loss + loss_weights[1] * keypoints_loss + loss_weights[2] * bbox_loss

            # Accumulate losses
            val_classification_loss += classification_loss.item()
            val_keypoint_loss += keypoints_loss.item()
            val_bbox_loss += bbox_loss.item()
            val_loss += total_loss.item()

            # Extract model outputs
            bbox, keypoints, workout_label = output

            # Calculate and accumulate accuracy metrics for workout classification
            workout_label_targets = torch.stack([target['workout_labels'] for target in new_targets]) 
            batch_TP, batch_FP, batch_FN = calculate_classification_accuracy(workout_label, workout_label_targets, num_classes)
            # Accumulate per-class TP, FP, FN
            total_TP += batch_TP
            total_FP += batch_FP
            total_FN += batch_FN
            # Calculate classification accuracy
            correct_predictions = (torch.argmax(workout_label, dim=1) == torch.argmax(workout_label_targets, dim=1)).sum().item()
            total_classification_correct += correct_predictions
            total_classification_count += len(workout_label_targets)

            # Calculate bbox accuracy
            bbox_targets = torch.stack([target['boxes'] for target in new_targets])
            bbox_from_heatmaps = extract_bbox_from_heatmaps(bbox)
            bbox_targets_from_heatmaps = extract_bbox_from_heatmaps(bbox_targets)
            bbox_accuracy = calculate_bbox_accuracy(bbox_from_heatmaps, bbox_targets_from_heatmaps)
            total_bbox_correct += bbox_accuracy * len(bbox_targets)
            total_bbox_count += len(bbox_targets)

            # Calculate keypoint accuracy
            keypoints_targets = torch.stack([target['keypoints'] for target in new_targets])
            keypoints_from_heatmaps = extract_keypoints_with_confidence(keypoints)
            keypoints_targets_from_heatmaps = extract_keypoints_with_confidence(keypoints_targets)
            correct_per_threshold, distance_sum, visible_count, predicted_visible_count = calculate_keypoint_accuracy(
                keypoints_from_heatmaps,
                keypoints_targets_from_heatmaps,
                thresholds=pck_thresholds, # PCK                
                image_size=resize_to          
            )
            for t in pck_thresholds:
                total_correct_per_threshold[t] += correct_per_threshold[t]

            total_distance_sum += distance_sum
            total_visible += visible_count
            total_predicted_visible += predicted_visible_count

            # Visualize predictions and targets for each epoch at batch 1 for the first 5 images
            if batch_idx == 0:
                batch_size = images.shape[0]  # Get batch size
                
                for i in range(batch_size):
                    # Convert image to numpy
                    sample_image = images[i].cpu().detach().numpy() 
                    
                    # Visualize keypoints and bounding boxes
                    vis_image = visualize_keypoints(
                        sample_image, 
                        keypoints_from_heatmaps[i].cpu().detach().numpy(), 
                        keypoints_targets_from_heatmaps[i].cpu().numpy(), 
                        sample_image.shape[2], 
                        sample_image.shape[1], 
                        bbox_from_heatmaps[i].cpu().detach().numpy(), 
                        bbox_targets_from_heatmaps[i].cpu().detach().numpy()
                    )

                    # Log the visualization to TensorBoard
                    writer.add_image(f'Validation_Visualization/Image_{i}', vis_image, num_epoch)
                    
                    # Log classification predictions and targets
                    # Predicted
                    probs = torch.nn.functional.softmax(workout_label[i], dim=0)
                    predicted_class_index = torch.argmax(probs, dim=0)
                    predicted_class_name = idx_to_class_name[predicted_class_index.item()]
                    predicted_prob = probs[predicted_class_index].item()
                    # Ground truth
                    true_class_index = torch.argmax(workout_label_targets[i], dim=0)
                    true_class_name = idx_to_class_name[true_class_index.item()]
                    # Log to TensorBoard
                    log_entry = f"Predicted: {predicted_class_name} (Prob: {predicted_prob:.4f})\nTrue: {true_class_name}"
                    writer.add_text(f"Validation_Classification/Image_{i}", log_entry, num_epoch)

                    # Log keypoints predictions and targets
                    # Convert heatmap to numpy and sum across channels to get a single channel heatmap
                    heatmap_pred = np.sum(keypoints[i].cpu().detach().numpy(), axis=0)
                    heatmap_target = np.sum(keypoints_targets[i].cpu().detach().numpy(), axis=0)
                    # Apply jet colormap (converts to RGB)
                    heatmap_pred_colored = cm.jet(heatmap_pred)[:, :, :3]  # Drop alpha channel
                    heatmap_target_colored = cm.jet(heatmap_target)[:, :, :3]
                    # Convert to tensor (C, H, W) 
                    heatmap_pred_tensor = torch.tensor(heatmap_pred_colored).permute(2, 0, 1)
                    heatmap_target_tensor = torch.tensor(heatmap_target_colored).permute(2, 0, 1)
                    # Log heatmaps in TensorBoard
                    writer.add_image(f"Validation_Keypoints_Heatmaps_Pred/{i}", heatmap_pred_tensor, num_epoch, dataformats="CHW")
                    writer.add_image(f"Validation_Keypoints_Heatmaps_Target/{i}", heatmap_target_tensor, num_epoch, dataformats="CHW")

                    # Log bounding box predictions and targets
                    # Normalize if necessary
                    bbox_pred = np.clip(bbox[i].cpu().detach().numpy(), 0, 1)
                    bbox_target = np.clip(bbox_targets[i].cpu().detach().numpy(), 0, 1)
                    # Apply jet colormap (convert to RGB)
                    bbox_pred_colored = cm.jet(bbox_pred)[:, :, :3]  # Drop alpha channel
                    bbox_target_colored = cm.jet(bbox_target)[:, :, :3]
                    # Convert to tensor (C, H, W)
                    bbox_pred_tensor = torch.tensor(bbox_pred_colored).permute(2, 0, 1)
                    bbox_target_tensor = torch.tensor(bbox_target_colored).permute(2, 0, 1)
                    # Log heatmaps in TensorBoard
                    writer.add_image(f"Validation_BBox_Heatmaps_Pred/{i}", bbox_pred_tensor, num_epoch, dataformats="CHW")
                    writer.add_image(f"Validation_BBox_Heatmaps_Target/{i}", bbox_target_tensor, num_epoch, dataformats="CHW")

    # Compute average validation loss
    avg_val_keypoint_loss = val_keypoint_loss / len(val_loader)
    avg_val_bbox_loss = val_bbox_loss / len(val_loader)
    avg_val_classification_loss = val_classification_loss / len(val_loader)
    avg_val_loss = val_loss / len(val_loader)
    # Log to TensorBoard
    writer.add_scalar("Epoch_Loss/Keypoint", avg_val_keypoint_loss, num_epoch) 
    writer.add_scalar("Epoch_Loss/BBox", avg_val_bbox_loss, num_epoch) 
    writer.add_scalar("Epoch_Loss/Classification", avg_val_classification_loss, num_epoch) 
    writer.add_scalar("Epoch_Loss/Total", avg_val_loss, num_epoch) 
    # Print to console
    print(f"Val Loss: {avg_val_loss:.6f}")
    print(f"Val Keypoint Loss: {avg_val_keypoint_loss:.6f}, Val BBox Loss: {avg_val_bbox_loss:.6f}, Val Classification Loss: {avg_val_classification_loss:.6f}")

    # Compute bbox accuracy at epoch level and log it
    val_bbox_accuracy = total_bbox_correct / total_bbox_count
    writer.add_scalar("Epoch_BBox_Accuracy/IoU", val_bbox_accuracy, num_epoch)

    # Compute PCK at all thresholds and log them
    for t in pck_thresholds:
        pck = total_correct_per_threshold[t] / (total_visible + 1e-6)
        writer.add_scalar(f"Epoch_Keypoint_Accuracy/PCK@{t}", pck, num_epoch)
    # Compute MPJPE and log it
    mpjpe = total_distance_sum / (total_visible + 1e-6)
    writer.add_scalar("Epoch_Keypoint_Accuracy/MPJPE", mpjpe, num_epoch)
    print(f"Val BBox Accuracy: {val_bbox_accuracy:.6f}, Val MPJPE: {mpjpe:.6f}")
    # Compute and log Predicted Visibility Ratio
    visibility_ratio = total_predicted_visible / (total_visible + 1e-6)
    writer.add_scalar("Epoch_Keypoint_Accuracy/Predicted_Visibility_Ratio", visibility_ratio, num_epoch)
    print(f"Predicted Visibility Ratio: {visibility_ratio:.4f}")

    # Compute per-class precision and recall at epoch level
    precision_per_class = total_TP / (total_TP + total_FP)
    precision_per_class[torch.isnan(precision_per_class)] = 0 
    recall_per_class = total_TP / (total_TP + total_FN)
    recall_per_class[torch.isnan(recall_per_class)] = 0  
    # Compute macro and weighted averages
    macro_precision = precision_per_class.mean().item() # Average across classes
    macro_recall = recall_per_class.mean().item() # Average across classes
    class_support = total_TP + total_FN  # Total actual instances per class
    weighted_precision = (precision_per_class * class_support).sum().item() / class_support.sum().item() # Weighted average precision
    weighted_recall = (recall_per_class * class_support).sum().item() / class_support.sum().item() # Weighted average recall

    # Compute final classification accuracy
    val_classification_accuracy = total_classification_correct / total_classification_count

    # Log Precision, Recall, and Accuracy
    writer.add_scalar("Epoch_Classification_Accuracy/Accuracy", val_classification_accuracy, num_epoch)
    writer.add_scalar("Epoch_Classification_Accuracy/Precision_Macro", macro_precision, num_epoch)
    writer.add_scalar("Epoch_Classification_Accuracy/Recall_Macro", macro_recall, num_epoch)
    writer.add_scalar("Epoch_Classification_Accuracy/Precision_Weighted", weighted_precision, num_epoch)
    writer.add_scalar("Epoch_Classification_Accuracy/Recall_Weighted", weighted_recall, num_epoch)

    # Log per-class precision and recall
    for cls in range(num_classes):
        class_name = idx_to_class_name[cls]
        writer.add_scalar(f"Epoch_Classification_Accuracy/Precision_{class_name}", precision_per_class[cls].item(), num_epoch)
        writer.add_scalar(f"Epoch_Classification_Accuracy/Recall_{class_name}", recall_per_class[cls].item(), num_epoch)

    # Close the TensorBoard writer
    writer.close()
