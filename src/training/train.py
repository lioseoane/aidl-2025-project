import os
import json
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_bbox_accuracy, calculate_keypoint_accuracy
from src.training.evaluate import evaluate_model
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps

def train_model(train_loader, model, class_name_to_idx, num_epochs=10, log_dir="logs/", checkpoint_dir="checkpoints", 
                val_loader=None, model_tag=None, autocast_enabled=True, loss_weights=[1, 1, 1], resize_to=[224, 224], 
                lr=[1e-4, 1e-3, 1e-4, 1e-3], pck_thresholds=[0.01, 0.05, 0.1]):

    # Clear cache
    torch.cuda.empty_cache()

    # Optimizer
    optimizer = optim.Adam([
        {'params': model.fpn.parameters(), 'lr': lr[0]}, 
        {'params': model.workout_label_head.parameters(), 'lr': lr[1]},  
        {'params': model.keypoints_head.parameters(), 'lr':lr[2]},
        {'params': model.bbox_head.parameters(), 'lr': lr[3]},
    ])

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping
    num_classes = len(idx_to_class_name)

    if model_tag is None:
        file_name = 'idx_to_class_name.json'
    else:
        file_name = f'{model_tag}.json'
    
    # Save the idx_to_class_name mapping
    with open(f'{file_name}', 'w') as f:
        json.dump(idx_to_class_name, f)

    model = model.to(device) # Move model to the same device as the data
    print(f"Using device: {device}")
    
    # Create logs folder and initialize TensorBoard writer
    log_dir_str = f'{log_dir}/{model_tag}/train_logs'
    os.makedirs(log_dir_str, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_str) # Initialize TensorBoard writer

    os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory

    # Initialize GradScaler for mixed precision
    if autocast_enabled == True:
        scaler = GradScaler()

    # Training loop
    for epoch in range(num_epochs):

        # Freeze workout parameters after 10 epochs
        if epoch == 10:
            for param in model.workout_label_head.parameters():
                param.requires_grad = False

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_classification_loss= 0.0
        running_keypoint_loss = 0.0
        running_bbox_loss = 0.0

        # Initialize accumulators for accuracy metrics at the epoch level
        total_classification_correct = 0
        total_classification_count = 0
        total_TP = torch.zeros(num_classes, dtype=torch.float32, device=device)
        total_FP = torch.zeros(num_classes, dtype=torch.float32, device=device)
        total_FN = torch.zeros(num_classes, dtype=torch.float32, device=device)
        total_bbox_correct = 0
        total_bbox_count = 0
        # Initialize totals for thresholds
        total_correct_per_threshold = {t: 0 for t in pck_thresholds}
        total_distance_sum = 0
        total_visible = 0
        total_predicted_visible = 0

        # Iterate over the training dataset
        for batch_idx, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # Move data to the same device as the model
            images = images.to(device)

            # Move targets to the same device as the model
            new_targets = []
            for i in range(len(targets["boxes"])):  # Iterating over the batch size (64)

                new_targets.append({
                    "boxes": targets["boxes"][i].to(device),  # Bounding box for image i
                    "workout_labels": targets["workout_labels"][i].to(device),  # Class label for image i
                    "keypoints": targets["heatmaps"][i].to(device),  # Keypoints for image i
                    "confidences": targets["confidences"][i].to(device),  # Keypoints for image i
                })

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass and compute loss
            if autocast_enabled:
                with autocast("cuda"): 
                    output = model(images) # Forward pass
                    bbox_loss, keypoint_loss, classification_loss = model.compute_losses(output, new_targets) # Compute loss
            else:
                output = model(images) # Forward pass
                bbox_loss, keypoint_loss, classification_loss = model.compute_losses(output, new_targets) # Compute loss

            # Compute total loss
            total_loss = loss_weights[0] * classification_loss + loss_weights[1] * keypoint_loss + loss_weights[2] * bbox_loss
            
            # Log batch loss to TensorBoard
            writer.add_scalar("Batch_Loss/Classification", classification_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch_Loss/Keypoint", keypoint_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch_Loss/BBox", bbox_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch_Loss/Total", total_loss.item(), epoch * len(train_loader) + batch_idx)

            # Backward pass
            if autocast_enabled:
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()

            # Accumulate loss for averaging
            running_classification_loss += classification_loss.item()
            running_keypoint_loss += keypoint_loss.item()
            running_bbox_loss += bbox_loss.item()
            running_loss += total_loss.item() 

            # Set model to evaluation mode, to disable dropout, batch normalization, etc.
            # Objective is to calculate the accuracy metrics at batch level
            model.eval()
            with torch.no_grad():
                bbox, keypoints, workout_label = output

                # Calculate and accumulate accuracy metrics for workout classification
                workout_label_targets = torch.stack([target['workout_labels'] for target in new_targets]) 
                batch_TP, batch_FP, batch_FN = calculate_classification_accuracy(workout_label, workout_label_targets, num_classes)
                # Accumulate true positives, false positives, and false negatives
                total_TP += batch_TP
                total_FP += batch_FP
                total_FN += batch_FN
                # Calculate classification accuracy
                correct_predictions = (torch.argmax(workout_label, dim=1) == torch.argmax(workout_label_targets, dim=1)).sum().item()
                total_classification_correct += correct_predictions
                total_classification_count += len(workout_label_targets)

                # Calculate bounding box accuracy
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

                # Visualize predictions and targets for each epoch at batch 1
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
                        writer.add_image(f'Train_Visualization/Image_{i}', vis_image, epoch)

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
                        writer.add_text(f"Train_Classification/Image_{i}", log_entry, epoch) 
                        
                        # Log keypoints predictions and targets
                        # Convert heatmap to numpy and sum across channels to get a single channel heatmap
                        heatmap_pred = np.sum(keypoints[i].cpu().detach().numpy(), axis=0)
                        heatmap_target = np.sum(keypoints_targets[i].cpu().detach().numpy(), axis=0)
                        # Apply jet colormap
                        heatmap_pred_colored = cm.jet(heatmap_pred)[:, :, :3] 
                        heatmap_target_colored = cm.jet(heatmap_target)[:, :, :3]
                        # Convert to tensor (C, H, W)
                        heatmap_pred_tensor = torch.tensor(heatmap_pred_colored).permute(2, 0, 1)
                        heatmap_target_tensor = torch.tensor(heatmap_target_colored).permute(2, 0, 1)
                        # Log to TensorBoard
                        writer.add_image(f"Train_Keypoints_Heatmaps_Pred/{i}", heatmap_pred_tensor, epoch, dataformats="CHW")
                        writer.add_image(f"Train_Keypoints_Heatmaps_Target/{i}", heatmap_target_tensor, epoch, dataformats="CHW") 

                        # Log bounding box predictions and targets
                        # Normalize if necessary
                        bbox_pred = np.clip(bbox[i].cpu().detach().numpy(), 0, 1)
                        bbox_target = np.clip(bbox_targets[i].cpu().detach().numpy(), 0, 1)
                        # Apply jet colormap 
                        bbox_pred_colored = cm.jet(bbox_pred)[:, :, :3] 
                        bbox_target_colored = cm.jet(bbox_target)[:, :, :3]
                        # Convert to tensor (C, H, W)
                        bbox_pred_tensor = torch.tensor(bbox_pred_colored).permute(2, 0, 1)
                        bbox_target_tensor = torch.tensor(bbox_target_colored).permute(2, 0, 1)
                        # Log heatmaps in TensorBoard
                        writer.add_image(f"Train_BBox_Heatmaps_Pred/{i}", bbox_pred_tensor, epoch, dataformats="CHW")
                        writer.add_image(f"Train_BBox_Heatmaps_Target/{i}", bbox_target_tensor, epoch, dataformats="CHW")
            
            # Set model back to training mode
            model.train()

        # Compute epoch loss and log it
        epoch_classification_loss = running_classification_loss / len(train_loader)
        epoch_keypoint_loss = running_keypoint_loss / len(train_loader)
        epoch_bbox_loss = running_bbox_loss / len(train_loader)
        epoch_loss = running_loss / len(train_loader)
        # Log to TensorBoard
        writer.add_scalar("Epoch_Loss/Classification", epoch_classification_loss, epoch)
        writer.add_scalar("Epoch_Loss/Keypoint", epoch_keypoint_loss, epoch)
        writer.add_scalar("Epoch_Loss/BBox", epoch_bbox_loss, epoch)
        writer.add_scalar("Epoch_Loss/Total", epoch_loss, epoch)
        # Print to console
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        print(f"  Classification Loss: {epoch_classification_loss:.6f}")
        print(f"  Keypoint Loss: {epoch_keypoint_loss:.6f}")
        print(f"  BBox Loss: {epoch_bbox_loss:.6f}")

        # Compute bbox accuracy at epoch level and log it
        epoch_bbox_accuracy = total_bbox_correct / total_bbox_count
        writer.add_scalar("Epoch_BBox_Accuracy/IoU", epoch_bbox_accuracy, epoch)

        # Compute PCK at all thresholds and log them
        for t in pck_thresholds:
            pck = total_correct_per_threshold[t] / (total_visible + 1e-6)
            writer.add_scalar(f"Epoch_Keypoint_Accuracy/PCK@{t}", pck, epoch)
        # Compute MPJPE and log it
        mpjpe = total_distance_sum / (total_visible + 1e-6)
        writer.add_scalar("Epoch_Keypoint_Accuracy/MPJPE", mpjpe, epoch)
        print(f"MPJPE: {mpjpe:.4f}")
        # Compute and log Predicted Visibility Ratio
        visibility_ratio = total_predicted_visible / (total_visible + 1e-6)
        writer.add_scalar("Epoch_Keypoint_Accuracy/Predicted_Visibility_Ratio", visibility_ratio, epoch)

        # Compute per-class precision and recall at epoch level
        precision_per_class = total_TP / (total_TP + total_FP)
        precision_per_class[torch.isnan(precision_per_class)] = 0  # Handle division by zero
        recall_per_class = total_TP / (total_TP + total_FN)
        recall_per_class[torch.isnan(recall_per_class)] = 0  # Handle division by zero
        # Compute macro and weighted averages
        macro_precision = precision_per_class.mean().item() # Average precision across all classes
        macro_recall = recall_per_class.mean().item() # Average recall across all classes
        class_support = total_TP + total_FN # Number of true positives + false negatives
        weighted_precision = (precision_per_class * class_support).sum().item() / class_support.sum().item() # Weighted average precision
        weighted_recall = (recall_per_class * class_support).sum().item() / class_support.sum().item() # Weighted average recall

        # Compute final classification accuracy
        classification_accuracy = total_classification_correct / total_classification_count

        # Log Precision, Recall, and Accuracy
        writer.add_scalar("Epoch_Classification_Accuracy/Accuracy", classification_accuracy, epoch)
        writer.add_scalar("Epoch_Classification_Accuracy/Precision_Macro", macro_precision, epoch)
        writer.add_scalar("Epoch_Classification_Accuracy/Recall_Macro", macro_recall, epoch)
        writer.add_scalar("Epoch_Classification_Accuracy/Precision_Weighted", weighted_precision, epoch)
        writer.add_scalar("Epoch_Classification_Accuracy/Recall_Weighted", weighted_recall, epoch)

        # Log per-class precision and recall
        for cls in range(num_classes):
            class_name = idx_to_class_name[cls]
            writer.add_scalar(f"Epoch_Classification_Accuracy/Precision_{class_name}", precision_per_class[cls].item(), epoch)
            writer.add_scalar(f"Epoch_Classification_Accuracy/Recall_{class_name}", recall_per_class[cls].item(), epoch)

        # Evaluate the model
        if val_loader != None:
            evaluate_model(val_loader, model, class_name_to_idx, model_tag=model_tag, num_epoch=epoch, loss_weights=loss_weights, pck_thresholds=pck_thresholds)

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Update the learning rate
        scheduler.step()

    # Close the TensorBoard writer
    writer.close()