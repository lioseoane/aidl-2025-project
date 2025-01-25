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

        for batch_idx, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Move data to the same device as the model
            images = images.to(device)
            new_targets = []
            for i in range(len(images)):
                target_dict = {
                    "boxes": targets["boxes"][i:i+1],
                    "labels": targets["labels"][i:i+1].view(-1),
                    "keypoints": targets["keypoints"][i:i+1],
                }
                for k, v in target_dict.items():
                    target_dict[k] = v.to(device)
                new_targets.append(target_dict)

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # print(f"Images shape: {images.shape}")
            # print(f"Batch size: {len(images)}")
            # for i, target in enumerate(new_targets):
            #     print(f"Target {i}:")
            #     for k, v in target.items():
            #         print(f"  Key: {k}, Value type: {type(v)}, Shape: {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")

            # Forward pass to calculate losses
            loss_dict = model(images, new_targets)  # During training, only losses are returned
            loss = sum(loss for loss in loss_dict.values())  # Total loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            running_loss += loss.item()  # Accumulate loss for averaging

            # Temporarily switch to evaluation mode to get predictions
            model.eval()
            with torch.no_grad():
                outputs = model(images)  # Get predictions
            model.train()  # Switch back to training mode

            # Print predictions for debugging
            # print(f"Outputs: {outputs}")  # Check the structure of the outputs
            # for i, output in enumerate(outputs):
            #     print(f"Prediction {i}:")
            #     for k, v in output.items():
            #         print(f"  Key: {k}, Value type: {type(v)}, Shape: {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")

            # Calculate PCK (Percentage of Correct Keypoints)
            if "keypoints" in outputs[0]:  # Ensure keypoints are in outputs
                pred_keypoints = torch.stack([out["keypoints"] for out in outputs])
                true_keypoints = torch.stack([target["keypoints"] for target in new_targets])

                pred_keypoints_np = pred_keypoints.cpu().detach().numpy().reshape(-1, 16, 2)
                true_keypoints_np = true_keypoints.cpu().detach().numpy().reshape(-1, 16, 2)
                pck_score = calculate_pck(pred_keypoints_np, true_keypoints_np, threshold=5)
                total_pck += pck_score
                total_samples += 1

            # Log batch loss and PCK to TensorBoard
            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            if total_samples > 0:
                writer.add_scalar("Batch/PCK", pck_score, epoch * len(train_loader) + batch_idx)

        # Compute epoch loss and log it
        epoch_loss = running_loss / len(train_loader)
        avg_pck = total_pck / total_samples if total_samples > 0 else 0
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        writer.add_scalar("Epoch/PCK", avg_pck, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, PCK: {avg_pck}%")

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Visualize predictions and targets every 5 epochs
        if epoch % 5 == 0 and "keypoints" in outputs[0]:
            sample_image = images[0].cpu().detach().numpy()
            predicted_keypoints = outputs[0]["keypoints"].cpu().detach().numpy().reshape(16, 2)
            true_keypoints = new_targets[0]["keypoints"].cpu().detach().numpy().reshape(16, 2)
            vis_image = visualize_keypoints(sample_image, predicted_keypoints, true_keypoints, sample_image.shape[1], sample_image.shape[2])
            writer.add_image('Keypoints/Visualization', vis_image, epoch)

    writer.close()
