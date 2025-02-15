import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_keypoint_accuracy, calculate_bbox_accuracy


def evaluate_model(val_loader, model, class_name_to_idx, log_dir="logs/val_logs", num_epoch=0):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if CUDA is available

    model = model.to(device)

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

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
            for i in range(len(targets["bbox"])):  # Iterating over the batch size (64)
                new_targets.append({
                    "bbox": targets["bbox"][i].to(device),  # Bounding box for image i
                    "workout_label": targets["workout_label"][i].to(device),  # Class label for image i
                    "keypoints": targets["keypoints"][i].to(device),  # Keypoints for image i
                })

            # Forward pass
            output = model(images) 
            losses = model.compute_losses(output, new_targets)
            loss_dict = {
                "classification_loss": losses[2],
                "bbox_loss": losses[0],
                "keypoint_loss": losses[1],
            }
            classification_loss = loss_dict["classification_loss"]
            keypoint_loss = loss_dict["keypoint_loss"]
            bbox_loss = loss_dict["bbox_loss"]

            total_loss = classification_loss + keypoint_loss + bbox_loss

            val_classification_loss += classification_loss.item()
            val_keypoint_loss += keypoint_loss.item()
            val_bbox_loss += bbox_loss.item()
            val_loss += total_loss.item()

             # Calculate overall accuracy for the epoch
            bbox, keypoints, workout_label = output

            # Calculate and accumulate accuracy metrics
            workout_label_targets = torch.stack([target['workout_label'] for target in new_targets]) 
            class_accuracy, batch_TP, batch_FP, batch_FN = calculate_classification_accuracy(workout_label, 
                                                                                              workout_label_targets, 
                                                                                              len(idx_to_class_name))
            total_classification_correct += class_accuracy * len(workout_label_targets)
            total_classification_count += len(workout_label_targets)
            total_classification_TP += batch_TP
            total_classification_FP += batch_FP
            total_classification_FN += batch_FN

            # Calculate bbox and keypoint accuracy
            bbox_targets = torch.stack([target['bbox'] for target in new_targets]) 
            bbox_accuracy = calculate_bbox_accuracy(bbox, bbox_targets)
            total_bbox_correct += bbox_accuracy * len(bbox_targets)
            total_bbox_count += len(bbox_targets)

            # Calculate keypoint accuracy
            keypoints_targets = torch.stack([target['keypoints'] for target in new_targets]) 
            keypoints_accuracy = calculate_keypoint_accuracy(keypoints, keypoints_targets)
            total_keypoints_correct += keypoints_accuracy * len(keypoints_targets)
            total_keypoints_count += len(keypoints_targets) 

            # Log batch loss to TensorBoard
            #writer.add_scalar("Validation_Batch_Loss/Keypoint", keypoint_loss.item(), batch_idx)
            #writer.add_scalar("Validation_Batch_Loss/BBox", bbox_loss.item(), batch_idx)
            #writer.add_scalar("Validation_Batch_Loss/Classification", classification_loss.item(), batch_idx)
            #writer.add_scalar("Validation_Batch_Loss/Total", total_loss.item(), batch_idx)

            # Visualize predictions and targets for each epoch at batch 1 for the first 5 images
            if batch_idx == 0:
                
                for i in range(8):

                    sample_image = images[i].cpu().detach().numpy() # Unfortunetely numpy doesn't work in CUDA
                    
                    # Visualize keypoints and bounding boxes
                    vis_image = visualize_keypoints(
                        sample_image, 
                        keypoints[i].cpu().detach().numpy(), 
                        keypoints_targets[i].cpu().numpy(), 
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
    writer.add_scalar("Validation_Accuracy/Keypoint_PCK", val_keypoints_accuracy, num_epoch)
    writer.add_scalar("Validation_Accuracy/BBox_IoU", val_bbox_accuracy, num_epoch)

    val_classification_precision = total_classification_TP / (total_classification_TP + total_classification_FP
                                                                    ) if (total_classification_TP + total_classification_FP) > 0 else 0.0
    val_classification_recall = total_classification_TP / (total_classification_TP + total_classification_FN
                                                                 ) if (total_classification_TP + total_classification_FN) > 0 else 0.0

    writer.add_scalar("Validation_Accuracy/Classification_Precision", val_classification_precision, num_epoch)
    writer.add_scalar("Validation_Accuracy/Classification_Recall", val_classification_recall, num_epoch)

    # Close the TensorBoard writer
    writer.close()
