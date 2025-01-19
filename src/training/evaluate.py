import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.metrics import calculate_pck
from src.utils.visualization import visualize_keypoints


def evaluate_model(val_loader, model, criterion, log_dir="logs/val_logs"):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Set model to evaluation mode
    model.eval()

    # Validation loop
    val_loss = 0.0
    total_pck = 0.0
    total_samples = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch_idx, (images, keypoints) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, keypoints = images.to(device), keypoints.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, keypoints.view(-1, 32))  # Assuming keypoints are flattened

            val_loss += loss.item()

            # Calculate PCK (Percentage of Correct Keypoints)
            pred_keypoints = outputs.cpu().detach().numpy().reshape(-1, 16, 2)  # reshape to keypoints format
            true_keypoints = keypoints.cpu().detach().numpy().reshape(-1, 16, 2)
            
            pck_score = calculate_pck(pred_keypoints, true_keypoints, threshold=5)  # using 5px as threshold
            total_pck += pck_score
            total_samples += 1

            # Log batch loss to TensorBoard
            writer.add_scalar("Batch/Validation_Loss", loss.item(), batch_idx)
            writer.add_scalar("Batch/PCK", pck_score, batch_idx)

            # Visualize keypoints on a sample image (every 10th batch)
            if batch_idx % 10 == 0:
                sample_image = images[0].cpu().detach().numpy()  # Take first image in batch
                predicted_keypoints = outputs[0].cpu().detach().numpy().reshape(16, 2)
                true_keypoints = keypoints[0].cpu().detach().numpy().reshape(16, 2)
                
                img_width, img_height = sample_image.shape[1], sample_image.shape[2]
                vis_image = visualize_keypoints(sample_image, predicted_keypoints, true_keypoints, img_width, img_height)
                writer.add_image('Keypoints/Visualization', vis_image, batch_idx)

    # Compute average validation loss and PCK
    avg_val_loss = val_loss / len(val_loader)
    avg_pck = total_pck / total_samples

    # Log average validation loss and PCK to TensorBoard
    writer.add_scalar("Validation/Loss", avg_val_loss, 0)  # Log average validation loss
    writer.add_scalar("Validation/PCK", avg_pck, 0)  # Log average PCK
    print(f"Average Validation Loss: {avg_val_loss}, Average PCK: {avg_pck}%")

    # Close the TensorBoard writer
    writer.close()
