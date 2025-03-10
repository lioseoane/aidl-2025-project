import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_bbox_accuracy, calculate_keypoint_average_precision
from src.training.evaluate import evaluate_model
from torch.amp import autocast, GradScaler
from datetime import datetime
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps
import matplotlib.cm as cm
import numpy as np
from torch.optim.lr_scheduler import StepLR



def train_model(train_loader, model, class_name_to_idx, num_epochs=10, log_dir="logs/train_logs", 
                checkpoint_dir="checkpoints", val_loader=None, model_save_path=None, autocast_enabled=True):

    # Clear cache
    torch.cuda.empty_cache()

    # Optimizer
    optimizer = optim.Adam([
        {'params': model.fpn.parameters(), 'lr': 1e-4},  # FPN and unfrozen layers
        {'params': model.workout_label_head.parameters(), 'lr': 4e-6},  
        {'params': model.keypoints_head.parameters(), 'lr': 1e-4},
        {'params': model.bbox_head.parameters(), 'lr': 1e-4},
    ])
    scheduler = StepLR(optimizer, step_size=80, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping
    num_classes = len(idx_to_class_name)

    model = model.to(device) # Move model to the same device as the data
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_str = f'{log_dir}/{model.backbone_label}_{timestamp}'
    os.makedirs(log_dir_str, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_str) # Initialize TensorBoard writer

    os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    for epoch in range(num_epochs):
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
        total_keypoints_correct = 0
        total_keypoints_count = 0

        # Iterate over the training dataset
        for batch_idx, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # Move data to the same device as the model
            images = images.to(device)

            # Move targets to the same device as the model
            # List range from 0 to batch size
            new_targets = []
            for i in range(len(targets["boxes"])):  # Iterating over the batch size (64)

                new_targets.append({
                    "boxes": targets["boxes"][i].to(device),  # Bounding box for image i
                    "workout_labels": targets["workout_labels"][i].to(device),  # Class label for image i
                    "keypoints": targets["heatmaps"][i].to(device),  # Keypoints for image i
                    "confidences": targets["confidences"][i].to(device),  # Keypoints for image i
                })

            #rpn_targets = []
            #for i in range(len(targets["boxes"])):
                #gt_boxes = targets["boxes"][i].to(device)

                # Fix shape if necessary
                #if gt_boxes.ndim == 1:
                    #gt_boxes = gt_boxes.unsqueeze(0)

                #rpn_targets.append({"boxes": gt_boxes})


            optimizer.zero_grad()  # Zero the gradients before backward pass

            if autocast_enabled:
                with autocast("cuda"):  # Autocast only when enabled
                    output = model(images)
                    bbox_loss, keypoint_loss, classification_loss = model.compute_losses(output, new_targets)
            else:
                output = model(images)  # Normal FP32
                bbox_loss, keypoint_loss, classification_loss = model.compute_losses(output, new_targets)

            total_loss = classification_loss + keypoint_loss + bbox_loss

            if autocast_enabled:
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()

            running_classification_loss += classification_loss.item()
            running_keypoint_loss += keypoint_loss.item()
            running_bbox_loss += bbox_loss.item()
            running_loss += total_loss.item()  # Accumulate loss for averaging

            model.eval()
            with torch.no_grad():
                # Calculate overall accuracy for the epoch
                bbox, keypoints, workout_label = output

                # Calculate and accumulate accuracy metrics
                workout_label_targets = torch.stack([target['workout_labels'] for target in new_targets]) 
                class_accuracy, batch_TP, batch_FP, batch_FN = calculate_classification_accuracy(workout_label, 
                                                                                                 workout_label_targets, 
                                                                                                 num_classes)

                total_TP += batch_TP
                total_FP += batch_FP
                total_FN += batch_FN
                correct_predictions = (torch.argmax(workout_label, dim=1) == torch.argmax(workout_label_targets, dim=1)).sum().item()
                total_classification_correct += correct_predictions
                total_classification_count += len(workout_label_targets)

                # Evaluate
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
                keypoints_accuracy = calculate_keypoint_average_precision(keypoints_from_heatmaps, keypoints_targets_from_heatmaps)
                total_keypoints_correct += keypoints_accuracy * len(keypoints_targets)
                total_keypoints_count += len(keypoints_targets) 

                # Log batch loss to TensorBoard
                writer.add_scalar("Batch_Loss/Classification", classification_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Batch_Loss/Keypoint", keypoint_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Batch_Loss/BBox", bbox_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Batch_Loss/Total", total_loss.item(), epoch * len(train_loader) + batch_idx)

                # Visualize predictions and targets for each epoch at batch 1 for the first 5 images
                if batch_idx == 0:
                    batch_size = images.shape[0]  # Get batch size
                    random_indices = torch.randperm(batch_size)[:4]  # Randomly pick 4 indices

                    for i, idx in enumerate(random_indices):

                        sample_image = images[i].cpu().detach().numpy() # Unfortunetely numpy doesn't work in CUDA
                        
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
                        writer.add_image(f'Visualization/Image_{i}', vis_image, epoch)

                        # Prediction
                        log_probs = torch.nn.functional.log_softmax(workout_label[i], dim=0)
                        predicted_class_index = torch.argmax(log_probs, dim=0)
                        predicted_class_name = idx_to_class_name[predicted_class_index.item()]
                        predicted_prob = torch.exp(log_probs[predicted_class_index.item()]).item()

                        # Ground truth
                        true_class_index = torch.argmax(workout_label_targets[i], dim=0)
                        true_class_name = idx_to_class_name[true_class_index.item()]

                        log_entry = f"Predicted: {predicted_class_name} (Prob: {predicted_prob:.4f})\nTrue: {true_class_name}"
                        writer.add_text(f"Classification/Image_{i}", log_entry, epoch)

                        # Convert heatmap to numpy and sum across channels
                        heatmap_pred = np.sum(keypoints[i].cpu().detach().numpy(), axis=0)
                        heatmap_target = np.sum(keypoints_targets[i].cpu().detach().numpy(), axis=0)

                        # Apply jet colormap (converts to RGB)
                        heatmap_pred_colored = cm.jet(heatmap_pred)[:, :, :3]  # Drop alpha channel
                        heatmap_target_colored = cm.jet(heatmap_target)[:, :, :3]

                        # Convert to tensor (C, H, W) format
                        heatmap_pred_tensor = torch.tensor(heatmap_pred_colored).permute(2, 0, 1)
                        heatmap_target_tensor = torch.tensor(heatmap_target_colored).permute(2, 0, 1)

                        # Log heatmaps in TensorBoard
                        writer.add_image(f"Heatmaps_Pred/{i}", heatmap_pred_tensor, epoch, dataformats="CHW")
                        writer.add_image(f"Heatmaps_Target/{i}", heatmap_target_tensor, epoch, dataformats="CHW")

                        # Normalize if necessary (values must be in [0, 1] for cm.jet)
                        bbox_pred = np.clip(bbox[i].cpu().detach().numpy(), 0, 1)
                        bbox_target = np.clip(bbox_targets[i].cpu().detach().numpy(), 0, 1)

                        # Apply jet colormap (convert to RGB)
                        bbox_pred_colored = cm.jet(bbox_pred)[:, :, :3]  # Drop alpha channel
                        bbox_target_colored = cm.jet(bbox_target)[:, :, :3]

                        # Convert to tensor (C, H, W)
                        bbox_pred_tensor = torch.tensor(bbox_pred_colored).permute(2, 0, 1)
                        bbox_target_tensor = torch.tensor(bbox_target_colored).permute(2, 0, 1)

                        # Log heatmaps in TensorBoard
                        writer.add_image(f"BBox_Heatmaps_Pred/{i}", bbox_pred_tensor, epoch, dataformats="CHW")
                        writer.add_image(f"BBox_Heatmaps_Target/{i}", bbox_target_tensor, epoch, dataformats="CHW")

            model.train()

        # Compute epoch loss and log it
        epoch_classification_loss = running_classification_loss / len(train_loader)
        epoch_keypoint_loss = running_keypoint_loss / len(train_loader)
        epoch_bbox_loss = running_bbox_loss / len(train_loader)
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Epoch_Loss/Classification", epoch_classification_loss, epoch)
        writer.add_scalar("Epoch_Loss/Keypoint", epoch_keypoint_loss, epoch)
        writer.add_scalar("Epoch_Loss/BBox", epoch_bbox_loss, epoch)
        writer.add_scalar("Epoch_Loss/Total", epoch_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        print(f"  Classification Loss: {epoch_classification_loss:.6f}")
        print(f"  Keypoint Loss: {epoch_keypoint_loss:.6f}")
        print(f"  BBox Loss: {epoch_bbox_loss:.6f}")

        # Compute epoch accuracies and log them
        epoch_bbox_accuracy = total_bbox_correct / total_bbox_count
        epoch_keypoints_accuracy = total_keypoints_correct / total_keypoints_count

        writer.add_scalar("Epoch_Accuracy/Keypoint_AP", epoch_keypoints_accuracy, epoch)
        writer.add_scalar("Epoch_Accuracy/BBox_IoU", epoch_bbox_accuracy, epoch)

        # Compute per-class precision and recall at epoch level
        precision_per_class = total_TP / (total_TP + total_FP)
        precision_per_class[torch.isnan(precision_per_class)] = 0  # Handle division by zero

        recall_per_class = total_TP / (total_TP + total_FN)
        recall_per_class[torch.isnan(recall_per_class)] = 0  # Handle division by zero

        # Compute macro and weighted averages
        macro_precision = precision_per_class.mean().item()
        macro_recall = recall_per_class.mean().item()

        class_support = total_TP + total_FN  # Total actual instances per class
        weighted_precision = (precision_per_class * class_support).sum().item() / class_support.sum().item()
        weighted_recall = (recall_per_class * class_support).sum().item() / class_support.sum().item()

        # Compute final classification accuracy
        classification_accuracy = total_classification_correct / total_classification_count

        # Log Precision, Recall, and Accuracy
        writer.add_scalar("Epoch_Accuracy/Classification_Accuracy", classification_accuracy, epoch)
        writer.add_scalar("Epoch_Accuracy/Classification_Precision_Macro", macro_precision, epoch)
        writer.add_scalar("Epoch_Accuracy/Classification_Recall_Macro", macro_recall, epoch)
        writer.add_scalar("Epoch_Accuracy/Classification_Precision_Weighted", weighted_precision, epoch)
        writer.add_scalar("Epoch_Accuracy/Classification_Recall_Weighted", weighted_recall, epoch)

        # Log per-class precision and recall
        for cls in range(num_classes):
            class_name = idx_to_class_name[cls]
            writer.add_scalar(f"Epoch_Accuracy/Precision_{class_name}", precision_per_class[cls].item(), epoch)
            writer.add_scalar(f"Epoch_Accuracy/Recall_{class_name}", recall_per_class[cls].item(), epoch)

        # Evaluate the model
        if val_loader != None:
            evaluate_model(val_loader, model, class_name_to_idx, num_epoch=epoch, timestamp=timestamp)

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        scheduler.step()

    writer.close()

    if model_save_path is None:
        file_name = 'idx_to_class_name.json'
    else:
        file_name = f'{model_save_path}.json'

    with open(f'{file_name}', 'w') as f:
        json.dump(idx_to_class_name, f)
