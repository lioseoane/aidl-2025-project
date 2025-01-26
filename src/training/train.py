import torch
from torch.utils.tensorboard import SummaryWriter  
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_keypoint_accuracy, calculate_bbox_accuracy
import os
from tqdm import tqdm


def train_model(train_loader, model, optimizer, num_epochs=10, log_dir="logs/train_logs", checkpoint_dir="checkpoints"):

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_class_accuracy = 0.0
        running_keypoint_accuracy = 0.0
        running_bbox_accuracy = 0.0
        total_samples = 0

        for batch_idx, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Move data to the same device as the model
            images = images.to(device)

            # Move targets to the same device as the model
            # List range from 0 to batch size
            new_targets = []
            for i in range(len(targets["boxes"])):  # Iterating over the batch size (64)
                new_targets.append({
                    "boxes": targets["boxes"][i].to(device),  # Bounding box for image i
                    "labels": targets["labels"][i].to(device),  # Class label for image i
                    "keypoints": targets["keypoints"][i].to(device),  # Keypoints for image i
                })

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass
            output = model(images) 
            
            # Now compute the losses separately
            loss_dict = model.compute_losses(output['boxes'], output['keypoints'], output['labels'], new_targets)
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            running_loss += loss.item()  # Accumulate loss for averaging

            # Compute accuracies
            batch_size = images.size(0)
            classification_accuracy = calculate_classification_accuracy(output['labels'], 
                                                                        [target['labels'].cpu().detach().numpy() for target in new_targets], 
                                                                        device=device)
            keypoint_accuracy = calculate_keypoint_accuracy(output['keypoints'].cpu().detach().numpy(), 
                                                            [target['keypoints'].cpu().detach().numpy() for target in new_targets], 
                                                            device=device)
            bbox_accuracy = calculate_bbox_accuracy(output['boxes'], 
                                                    [target['boxes'] for target in new_targets], 
                                                    device=device)

            # Accumulate accuracies
            running_class_accuracy += classification_accuracy
            running_keypoint_accuracy += keypoint_accuracy
            running_bbox_accuracy += bbox_accuracy
            total_samples += batch_size

            model.eval()
            with torch.no_grad():
                outputs = model(images)  # Get predictions
            model.train()  

            # Log batch loss to TensorBoard
            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(train_loader) + batch_idx)

        # Compute epoch loss and log it
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)

        # Compute epoch accuracies and log them
        epoch_class_accuracy = running_class_accuracy / total_samples
        epoch_keypoint_accuracy = running_keypoint_accuracy / total_samples
        epoch_bbox_accuracy = running_bbox_accuracy / total_samples
        writer.add_scalar("Epoch/ClassificationAccuracy", epoch_class_accuracy, epoch)
        writer.add_scalar("Epoch/KeypointAccuracy", epoch_keypoint_accuracy, epoch)
        writer.add_scalar("Epoch/BboxAccuracy", epoch_bbox_accuracy, epoch)

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Visualize predictions and targets every 5 epochs
        if epoch % 1 == 0:

            for i in range(min(3, len(images))):
                sample_image = images[i].cpu().detach().numpy()

                # Get the predicted and true keypoints
                predicted_keypoints = outputs["keypoints"][i].cpu().detach().numpy()
                true_keypoints = new_targets[i]["keypoints"].cpu().detach().numpy()

                # Get the predicted and true bounding boxes
                predicted_bbox = outputs['boxes'][i].cpu().detach().numpy().flatten()
                true_bbox = new_targets[i]['boxes'].cpu().detach().numpy().flatten()

                # Visualize keypoints and bounding boxes
                vis_image = visualize_keypoints(
                    sample_image, 
                    predicted_keypoints, 
                    true_keypoints, 
                    sample_image.shape[1], 
                    sample_image.shape[2], 
                    predicted_bbox, 
                    true_bbox
                )

                # Log the visualization to TensorBoard
                writer.add_image(f'Keypoints_and_Bboxes/Visualization_{i}', vis_image, epoch)

                # Get the predicted class and true class
                predicted_labels = outputs['labels'][i] 
                true_labels = new_targets[i]['labels'] 

                predicted_class = torch.argmax(predicted_labels).item()
                true_class = true_labels.item()
                
                # Log bounding box and class label information to TensorBoard
                writer.add_scalar(f"PredictedClass/{i}", predicted_class, epoch)
                writer.add_scalar(f"TrueClass/{i}", true_class, epoch)

    writer.close()
