import torch
from torch.utils.tensorboard import SummaryWriter  
from src.utils.metrics import calculate_pck
from src.utils.visualization import visualize_keypoints
import os
from tqdm import tqdm


def train_model(train_loader, model, optimizer, criterion, num_epochs=10, log_dir="logs/train_logs", checkpoint_dir="checkpoints"):

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
        total_pck = 0.0
        total_samples = 0
        
        for batch_idx, (images, keypoints) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Move data to the same device as the model
            images, keypoints = images.to(device), keypoints.to(device)

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass
            outputs = model(images)  # Get predictions from the model
            loss = criterion(outputs, keypoints.view(-1, 32))  # assuming keypoints are of size 32 (x, y for each keypoint)

            # Backward pass and optimization
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            running_loss += loss.item()  # Accumulate loss for averaging
            
            # Calculate PCK (Percentage of Correct Keypoints)
            pred_keypoints = outputs.cpu().detach().numpy().reshape(-1, 16, 2)  # reshape to keypoints format
            true_keypoints = keypoints.cpu().detach().numpy().reshape(-1, 16, 2)
            
            pck_score = calculate_pck(pred_keypoints, true_keypoints, threshold=5)  # using 5px as threshold
            total_pck += pck_score
            total_samples += 1
            
            # Log batch loss and PCK to TensorBoard
            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch/PCK", pck_score, epoch * len(train_loader) + batch_idx)
        
        # Compute epoch loss and log it
        epoch_loss = running_loss / len(train_loader)
        avg_pck = total_pck / total_samples
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        writer.add_scalar("Epoch/PCK", avg_pck, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, PCK: {avg_pck}%")

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Visualize the predictions and ground truth keypoints on a sample image
        if epoch % 5 == 0:  # Every 5 epochs, log a sample visualization
            sample_image = images[0].cpu().detach().numpy()  # Take first image in batch
            predicted_keypoints = outputs[0].cpu().detach().numpy().reshape(16, 2)
            true_keypoints = keypoints[0].cpu().detach().numpy().reshape(16, 2)
            
            img_width, img_height = sample_image.shape[1], sample_image.shape[2]
            vis_image = visualize_keypoints(sample_image, predicted_keypoints, true_keypoints, img_width, img_height)
            writer.add_image('Keypoints/Visualization', vis_image, epoch)

    writer.close()