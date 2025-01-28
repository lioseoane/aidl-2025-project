import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_keypoint_accuracy, calculate_bbox_accuracy


def train_model(train_loader, model, class_name_to_idx, num_epochs=10, log_dir="logs/train_logs", checkpoint_dir="checkpoints"):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Initialize optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping

    model = model.to(device) # Move model to the same device as the data
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=log_dir) # Initialize TensorBoard writer

    os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Initialize accumulators for accuracy metrics at the epoch level
        total_classification_correct = 0
        total_classification_count = 0
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
            for i in range(len(targets["bbox"])):  # Iterating over the batch size (64)
                new_targets.append({
                    "bbox": targets["bbox"][i].to(device),  # Bounding box for image i
                    "workout_label": targets["workout_label"][i].to(device),  # Class label for image i
                    "keypoints": targets["keypoints"][i].to(device),  # Keypoints for image i
                })

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass
            output = model(images) 
            losses = model.compute_losses(output, new_targets)
            loss = sum(loss for loss in losses)

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            running_loss += loss.item()  # Accumulate loss for averaging

            # Calculate overall accuracy for the epoch
            bbox, keypoints, workout_label = output

            # Calculate and accumulate accuracy metrics
            workout_label_targets = torch.stack([target['workout_label'] for target in new_targets]) 
            classification_accuracy = calculate_classification_accuracy(workout_label, workout_label_targets)
            total_classification_correct += classification_accuracy * len(workout_label_targets)
            total_classification_count += len(workout_label_targets)

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
            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(train_loader) + batch_idx)

            # Visualize predictions and targets for each epoch at batch 1 for the first 5 images
            if batch_idx == 0:
                for i in range(4):

                    sample_image = images[i].cpu().detach().numpy() # Unfortunetely numpy doesn't work in CUDA
                    
                    # Visualize keypoints and bounding boxes
                    vis_image = visualize_keypoints(
                        sample_image, 
                        keypoints[i].cpu().detach().numpy(), 
                        keypoints_targets[i].cpu().numpy(), 
                        sample_image.shape[1], 
                        sample_image.shape[2], 
                        bbox[i].squeeze().cpu().detach().numpy(), 
                        bbox_targets[i].cpu().detach().numpy()
                    )

                    # Log the visualization to TensorBoard
                    writer.add_image(f'Keypoints_and_Bboxes/Visualization_{i}', vis_image, epoch)
                    
                    predicted_indices = torch.argmax(workout_label[i], dim=0)
                    predicted_class_names = idx_to_class_name[predicted_indices.item()]
                    true_indices = torch.argmax(workout_label_targets[i], dim=0) 
                    true_class_names = idx_to_class_name[true_indices.item()]
                    writer.add_text(f"PredictedClass/{i}", predicted_class_names, epoch)
                    writer.add_text(f"TrueClass/{i}", true_class_names, epoch)

        # Compute epoch loss and log it
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)

        # Compute epoch accuracies and log them
        epoch_classification_accuracy = total_classification_correct / total_classification_count
        epoch_bbox_accuracy = total_bbox_correct / total_bbox_count
        epoch_keypoints_accuracy = total_keypoints_correct / total_keypoints_count

        writer.add_scalar("Epoch/ClassificationAccuracy", epoch_classification_accuracy, epoch)
        writer.add_scalar("Epoch/KeypointAccuracy", epoch_keypoints_accuracy, epoch)
        writer.add_scalar("Epoch/BboxAccuracy", epoch_bbox_accuracy, epoch)

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    writer.close()
