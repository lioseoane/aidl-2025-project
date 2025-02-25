import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_keypoint_accuracy, calculate_bbox_accuracy, calculate_keypoint_average_precision
from torch.amp import autocast
import os
from src.utils.heatmaps import extract_keypoints_with_confidence
import matplotlib.cm as cm
import numpy as np

def evaluate_model(val_loader, model, class_name_to_idx, log_dir="logs/val_logs", num_epoch=0, timestamp=''):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if CUDA is available

    model = model.to(device)

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping

    # Initialize TensorBoard writer
    log_dir_str = f'{log_dir}/{model.backbone_label}_{timestamp}'
    os.makedirs(log_dir_str, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_str)

    # Set model to evaluation mode
    model.eval()

    # Validation loop
    val_classification_loss= 0.0
    val_keypoint_loss = 0.0
    val_bbox_loss = 0.0
    val_loss = 0.0

    # Initialize accumulators for accuracy metrics at the epoch level
    total_classification_correct = 0
    total_classification_count = 0
    total_classification_TP = 0
    total_classification_FP = 0
    total_classification_FN = 0

    total_bbox_correct = 0
    total_bbox_count = 0
    total_keypoints_correct = 0
    total_keypoints_count = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch_idx, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):

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
                })

            # Forward pass and Losses
            with autocast("cuda"):  # Automatically uses FP16 where it can
                output = model(images) 
                keypoints_loss, boxes_loss, classification_loss = model.compute_losses(output, new_targets) # Losses

            loss_dict = {
                "classification_loss": classification_loss,
                "boxes_loss": boxes_loss,
                "keypoints_loss": keypoints_loss,
            }

            classification_loss = loss_dict["classification_loss"]
            keypoint_loss = loss_dict["keypoints_loss"]
            bbox_loss = loss_dict["boxes_loss"]

            total_loss = classification_loss + keypoint_loss + bbox_loss

            val_classification_loss += classification_loss.item()
            val_keypoint_loss += keypoint_loss.item()
            val_bbox_loss += bbox_loss.item()
            val_loss += total_loss.item()

            # Calculate overall accuracy for the epoch
            bbox, keypoints, workout_label = output

            # Calculate and accumulate accuracy metrics
            workout_label_targets = torch.stack([target['workout_labels'] for target in new_targets]) 
            class_accuracy, batch_TP, batch_FP, batch_FN = calculate_classification_accuracy(workout_label, 
                                                                                             workout_label_targets, 
                                                                                             len(idx_to_class_name))
            total_classification_correct += class_accuracy * len(workout_label_targets)
            total_classification_count += len(workout_label_targets)
            total_classification_TP += batch_TP
            total_classification_FP += batch_FP
            total_classification_FN += batch_FN

            # Calculate bbox accuracy
            bbox_targets = torch.stack([target['boxes'] for target in new_targets])
            bbox_accuracy = calculate_bbox_accuracy(bbox, bbox_targets)
            total_bbox_correct += bbox_accuracy * len(bbox_targets)
            total_bbox_count += len(bbox_targets)

            # Calculate keypoint accuracy
            keypoints_targets = torch.stack([target['keypoints'] for target in new_targets])
            keypoints_from_heatmaps = extract_keypoints_with_confidence(keypoints)
            keypoints_targets_from_heatmaps = extract_keypoints_with_confidence(keypoints_targets)
            keypoints_accuracy = calculate_keypoint_average_precision(keypoints_from_heatmaps, keypoints_targets_from_heatmaps)
            total_keypoints_correct += keypoints_accuracy * len(keypoints_targets)
            total_keypoints_count += len(keypoints_targets) 

            # Visualize predictions and targets for each epoch at batch 1 for the first 5 images
            if batch_idx == 0:
                
                for i in range(4):

                    sample_image = images[i].cpu().detach().numpy() # Unfortunetely numpy doesn't work in CUDA
                    
                    # Visualize keypoints and bounding boxes
                    vis_image = visualize_keypoints(
                        sample_image, 
                        keypoints_from_heatmaps[i].cpu().detach().numpy(), 
                        keypoints_targets_from_heatmaps[i].cpu().numpy(), 
                        sample_image.shape[2], 
                        sample_image.shape[1], 
                        bbox[i].squeeze().cpu().detach().numpy(), 
                        bbox_targets[i].cpu().detach().numpy()
                    )

                    # Log the visualization to TensorBoard
                    writer.add_image(f'Validation_Visualization/Image_{i}', vis_image, num_epoch)
                    
                    # Prediction
                    log_probs = torch.nn.functional.log_softmax(workout_label[i], dim=0)
                    predicted_class_index = torch.argmax(log_probs, dim=0)
                    predicted_class_name = idx_to_class_name[predicted_class_index.item()]
                    predicted_prob = torch.exp(log_probs[predicted_class_index.item()]).item()

                    # Ground truth
                    true_class_index = torch.argmax(workout_label_targets[i], dim=0)
                    true_class_name = idx_to_class_name[true_class_index.item()]

                    log_entry = f"Predicted: {predicted_class_name} (Prob: {predicted_prob:.4f})\nTrue: {true_class_name}"
                    writer.add_text(f"Validation_Classification/Image_{i}", log_entry, num_epoch)

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
                    writer.add_image(f"Heatmaps_Pred/{i}", heatmap_pred_tensor, num_epoch, dataformats="CHW")
                    writer.add_image(f"Heatmaps_Target/{i}", heatmap_target_tensor, num_epoch, dataformats="CHW")

    # Compute average validation loss
    avg_val_keypoint_loss = val_keypoint_loss / len(val_loader)
    avg_val_bbox_loss = val_bbox_loss / len(val_loader)
    avg_val_classification_loss = val_classification_loss / len(val_loader)
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("Validation_Loss/Keypoint", avg_val_keypoint_loss, num_epoch) 
    writer.add_scalar("Validation_Loss/BBox", avg_val_bbox_loss, num_epoch) 
    writer.add_scalar("Validation_Loss/Classification", avg_val_classification_loss, num_epoch) 
    writer.add_scalar("Validation_Loss/Total", avg_val_loss, num_epoch) 

     # Compute epoch accuracies and log them
    val_classification_accuracy = total_classification_correct / total_classification_count
    val_bbox_accuracy = total_bbox_correct / total_bbox_count
    val_keypoints_accuracy = total_keypoints_correct / total_keypoints_count

    writer.add_scalar("Validation_Accuracy/Classification_Accuracy", val_classification_accuracy, num_epoch)
    writer.add_scalar("Validation_Accuracy/Keypoint_AP", val_keypoints_accuracy, num_epoch)
    writer.add_scalar("Validation_Accuracy/BBox_IoU", val_bbox_accuracy, num_epoch)

    val_classification_precision = total_classification_TP / (total_classification_TP + total_classification_FP
                                                                    ) if (total_classification_TP + total_classification_FP) > 0 else 0.0
    val_classification_recall = total_classification_TP / (total_classification_TP + total_classification_FN
                                                                 ) if (total_classification_TP + total_classification_FN) > 0 else 0.0

    #writer.add_scalar("Validation_Accuracy/Classification_Precision", val_classification_precision, num_epoch)
    #writer.add_scalar("Validation_Accuracy/Classification_Recall", val_classification_recall, num_epoch)

    # Close the TensorBoard writer
    writer.close()
